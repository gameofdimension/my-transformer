import math

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import get_activation

from model.common import attention_func
from model.llama_config import LlamaConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        val = hidden_states * hidden_states
        norm = torch.sqrt(torch.mean(val, dim=-1, keepdim=True) + self.eps)
        return hidden_states / norm * self.weight


class Rotary:
    def __init__(self, d: int):
        assert d % 2 == 0
        self.d = d
        self.matrix_lst = []

    def _pad(self, target: int):
        base = 10000
        for m in range(len(self.matrix_lst) + 1, target + 1):
            matrix = torch.zeros(size=(self.d, self.d))

            for j in range(self.d // 2):
                theta = base ** (-2 * (j - 1) / self.d)
                matrix[2 * j, 2 * j] = math.cos(m * theta)
                matrix[2 * j, 2 * j + 1] = -math.sin(m * theta)
                matrix[2 * j + 1, 2 * j + 1] = math.cos(m * theta)
                matrix[2 * j + 1, 2 * j - 1] = math.sin(m * theta)
            assert m == len(self.matrix_lst) + 1
            self.matrix_lst.append(matrix)

    def apply(self, m: int, vec: torch.Tensor):
        assert m >= 1
        assert vec.size(-1) == self.d
        if m > len(self.matrix_lst):
            self._pad(m)
        matrix = self.matrix_lst[m - 1]
        return matrix @ vec


class Mlp(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor):
        activation = self.act_fn(self.gate_proj(hidden_states))
        hidden_states = activation * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.q_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.k_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.v_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.o_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

        self.config = config
        self.rotary = Rotary(self.config.hidden_size // self.config.num_attention_heads)

    def forward(self, hidden_states: torch.Tensor):
        # [seq_length, hidden_size]
        assert len(hidden_states.size()) == 2
        all_q = self.q_proj(hidden_states)
        all_k = self.k_proj(hidden_states)
        all_v = self.v_proj(hidden_states)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        def get_q(idx: int, head: int):
            return self.rotary.apply(idx + 1, all_q[idx, head * head_dim:(head + 1) * head_dim])

        def get_k(idx: int, head: int):
            return self.rotary.apply(idx + 1, all_k[idx, head * head_dim:(head + 1) * head_dim])

        def get_v(idx: int, head: int):
            return all_v[idx, head * head_dim:(head + 1) * head_dim]

        seq_length = hidden_states.shape[0]
        output = attention_func(
            seq_length=seq_length, num_attention_heads=self.config.num_attention_heads,
            hidden_size=self.config.hidden_size, get_q=get_q, get_k=get_k, get_v=get_v)
        return self.o_proj(output)


class Block(nn.Module):
    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rn1 = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mha = MultiHeadAttention(config)

        self.rn2 = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Mlp(config)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        for f, g in zip([self.mha, self.mlp], [self.rn1, self.rn2]):
            gx = g(x)
            fx = f(gx)
            x = x + fx
        return x


def name_mapping(param: str):
    out = {
        "word_embedding_table.weight": "model.embed_tokens.weight",
        "rms.weight": "model.norm.weight",
    }
    if param in out:
        return out[param]

    li = param.split('.')[1]
    prefix = f"model.layers.{li}."
    if "rn1.weight" in param:
        postfix = "input_layernorm.weight"
    elif "mha.q_proj.weight" in param:
        postfix = "self_attn.q_proj.weight"
    elif "mha.k_proj.weight" in param:
        postfix = "self_attn.k_proj.weight"
    elif "mha.v_proj.weight" in param:
        postfix = "self_attn.v_proj.weight"
    elif "mha.o_proj.weight" in param:
        postfix = "self_attn.o_proj.weight"
    elif "rn2.weight" in param:
        postfix = "post_attention_layernorm.weight"
    elif "mlp.gate_proj.weight" in param:
        postfix = "mlp.gate_proj.weight"
    elif "mlp.up_proj.weight" in param:
        postfix = "mlp.up_proj.weight"
    elif "mlp.down_proj.weight" in param:
        postfix = "mlp.down_proj.weight"
    else:
        assert False

    return prefix + postfix


class Model(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.rms = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.word_embedding_table(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.rms(hidden_states), layers_output

    def load_weights_from_hf(self, model_id):
        """
        因为是复刻 huggingface gpt2，所以可以直接加载其模型权重
        :return:
        """
        # model_id = 'felixdae/Llama-2-7b-hf'
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)
