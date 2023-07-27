import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import get_activation

from model.baichuan13b_config import Baichuan13bConfig
from model.common import RMSNorm, attention_func


class Mlp(nn.Module):
    def __init__(self, config: Baichuan13bConfig):
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
    def __init__(self, config: Baichuan13bConfig):
        super().__init__()
        self.W_pack = nn.Linear(in_features=config.hidden_size, out_features=3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        all_head_qkv = self.W_pack(hidden_states)
        dim = self.config.hidden_size // self.config.num_attention_heads
        step = all_head_qkv.size(dim=-1) // 3

        def get_q(idx: int, head: int):
            base = step * 0
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        def get_k(idx: int, head: int):
            base = step * 1
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        def get_v(idx: int, head: int):
            base = step * 2
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        seq_length = hidden_states.shape[0]
        output = attention_func(
            seq_length=seq_length, num_attention_heads=self.config.num_attention_heads,
            hidden_size=self.config.hidden_size, get_q=get_q, get_k=get_k, get_v=get_v)
        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, config: Baichuan13bConfig):
        super().__init__()

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
        "embed_tokens.weight": "model.embed_tokens.weight",
        "rms.weight": "model.norm.weight",
    }
    if param in out:
        return out[param]

    li = param.split('.')[1]
    prefix = f"model.layers.{li}."
    if "rn1.weight" in param:
        postfix = "input_layernorm.weight"
    elif "mha.W_pack.weight" in param:
        postfix = "self_attn.W_pack.weight"
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
    def __init__(self, config: Baichuan13bConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.rms = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.embed_tokens(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.rms(hidden_states), layers_output

    def load_weights_from_hf(self, model_id):
        # model_id = "felixdae/Baichuan-13B-Chat"
        ref_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)
