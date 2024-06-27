import torch
from torch import nn
from transformers import AutoModelForCausalLM

from model.common import RMSNorm, apply_rotary, precompute_cos_sin
from model.qwen_config import QwenConfig


class MLP(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        ff_dim_in = config.intermediate_size // 2
        self.w1 = nn.Linear(
            config.hidden_size, ff_dim_in, bias=False,
        )
        self.w2 = nn.Linear(
            config.hidden_size, ff_dim_in, bias=False,
        )
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * nn.functional.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class SelfAttention(nn.Module):
    def __init__(self, config: QwenConfig, get_cos_sin):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        cos_sin = get_cos_sin()
        self.cos = cos_sin[0]
        self.sin = cos_sin[1]

    def forward(self, hidden_states: torch.Tensor):
        assert len(hidden_states.size()) == 3
        bsz, q_len, hsz = hidden_states.size()
        assert hsz == self.hidden_size

        mixed_x_layer = self.c_attn(hidden_states)
        query, key, value = mixed_x_layer.split(self.hidden_size, dim=2)

        all_q = query.view(
            bsz, q_len, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        all_k = key.view(
            bsz, q_len, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        all_v = value.view(
            bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        cos, sin = self.cos[:q_len], self.sin[:q_len]
        all_q = apply_rotary(all_q, cos, sin).permute(1, 2, 0, 3)
        all_k = apply_rotary(all_k, cos, sin).permute(1, 2, 0, 3)

        out = nn.functional.scaled_dot_product_attention(
            query=all_q, key=all_k, value=all_v, is_causal=True)

        out = out.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)
        out = self.c_proj(out)
        return out


class Block(nn.Module):
    def __init__(
            self, config: QwenConfig,
            get_cos_sin):
        super().__init__()
        hidden_size = config.hidden_size

        self.ln_1 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attn = SelfAttention(config, get_cos_sin=get_cos_sin)
        self.ln_2 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def name_mapping(param: str):
    return "transformer."+param


class Model(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.wte = nn.Embedding(
            config.vocab_size, config.hidden_size)

        get_cos_sin = precompute_cos_sin(
            config.rotary_emb_base,
            config.seq_length,
            config.hidden_size//config.num_attention_heads,
            config.device)
        self.h = nn.ModuleList(
            [
                Block(config, get_cos_sin)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.wte(input_ids)
        layers_output = []
        for layer in self.h:
            layers_output.append(hidden_states.detach())
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        layers_output.append(hidden_states.detach())
        return hidden_states, layers_output

    def load_weights_from_hf(self, ref_model, model_id):
        """
        :return:
        """
        # model_id = "Qwen/Qwen-1_8B"
        if ref_model is None:
            ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)
