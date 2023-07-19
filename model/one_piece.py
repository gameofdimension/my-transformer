from typing import List

import torch
from torch import nn
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM
from transformers.activations import get_activation

from model.config import Gpt2Config


class Mlp(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_fn = get_activation(config.activation_function)

        self.up_proj = nn.Linear(in_features=config.n_embd, out_features=4 * config.n_embd)
        self.down_proj = nn.Linear(in_features=4 * config.n_embd, out_features=config.n_embd)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        return self.down_proj(self.action_fn(self.up_proj(x)))


def weighted_sum(scores: List[torch.Tensor], values: List[torch.Tensor]) -> torch.Tensor:
    v = torch.stack(values)
    scores = torch.Tensor(scores)
    probs = softmax(scores, dim=-1)
    return probs @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config.n_embd % config.n_head == 0
        dim = config.n_embd // config.n_head
        self.qkv_dim = 3 * dim
        self.qkv_proj = nn.Linear(config.n_embd, self.qkv_dim * config.n_head)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """

        all_head_qkv = self.qkv_proj(hidden_states)
        dim = self.qkv_dim // 3
        step = all_head_qkv.size(dim=-1) // 3

        def get_q(idx: int, head: int):
            # base = 3 * dim * head
            base = step * 0
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        def get_k(idx: int, head: int):
            base = step * 1
            # base = 3 * dim * head
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        def get_v(idx: int, head: int):
            # base = 3 * dim * head
            base = step * 2
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        seq_length = hidden_states.shape[0]
        scale = dim ** 0.5
        output = []
        for tk in range(seq_length):
            final_value = []
            for h in range(self.config.n_head):
                q = get_q(tk, h)
                s = []
                v = []
                for p in range(1 + tk):
                    s.append(torch.dot(q, get_k(p, h)) / scale)
                    v.append(get_v(p, h))
                final_value.append(weighted_sum(s, v))
            final_value = torch.stack(final_value).view(1, -1).squeeze()
            assert final_value.shape[0] == self.config.n_embd
            output.append(final_value)
        output = torch.stack(output)
        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ln1 = nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mha = MultiHeadAttention(config)

        self.ln2 = nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = Mlp(config)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        for f, g in zip([self.mha, self.mlp], [self.ln1, self.ln2]):
            gx = g(x)
            fx = f(gx)
            x = x + fx
        return x


def name_mapping(param: str):
    out = {
        "word_embedding_table.weight": "transformer.wte.weight",
        "position_embedding_table.weight": "transformer.wpe.weight",
        "ln.weight": "transformer.ln_f.weight",
        "ln.bias": "transformer.ln_f.bias",
    }
    if param in out:
        return out[param]

    li = param.split('.')[1]
    prefix = f"transformer.h.{li}."
    if "ln1.weight" in param:
        postfix = "ln_1.weight"
    elif "ln1.bias" in param:
        postfix = "ln_1.bias"
    elif "mha.qkv_proj.weight" in param:
        postfix = "attn.c_attn.weight"
    elif "mha.qkv_proj.bias" in param:
        postfix = "attn.c_attn.bias"
    elif "mha.out_proj.weight" in param:
        postfix = "attn.c_proj.weight"
    elif "mha.out_proj.bias" in param:
        postfix = "attn.c_proj.bias"
    elif "ln2.weight" in param:
        postfix = "ln_2.weight"
    elif "ln2.bias" in param:
        postfix = "ln_2.bias"
    elif "mlp.up_proj.weight" in param:
        postfix = "mlp.c_fc.weight"
    elif "mlp.up_proj.bias" in param:
        postfix = "mlp.c_fc.bias"
    elif "down_proj.weight" in param:
        postfix = "mlp.c_proj.weight"
    elif "down_proj.bias" in param:
        postfix = "mlp.c_proj.bias"
    else:
        assert False

    return prefix + postfix


class Model(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        """
        单序列玩具版 gpt2，纯手搓
        不支持 batch 输入
        :param input_ids: [seq_length]
        :return:
        """
        assert len(input_ids.shape) == 1
        position_ids = torch.LongTensor(range(input_ids.shape[0]))

        word_embeddings = self.word_embedding_table(input_ids)
        position_embeddings = self.position_embedding_table(position_ids)

        hidden_states = word_embeddings + position_embeddings
        layers_output = [hidden_states.detach()]

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())

        return self.ln(hidden_states), layers_output

    def load_weights_from_hf(self):
        """
        因为是复刻 huggingface gpt2，所以可以直接加载其模型权重
        :return:
        """
        model_id = 'gpt2'
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            if "weight" in name and ('.mlp.' in name or '.mha.' in name):
                param.data.copy_(ref_param.transpose(0, 1))
            else:
                param.data.copy_(ref_param)
