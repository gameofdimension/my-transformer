import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN

from model.common import RMSNorm, apply_rotary, precompute_cos_sin
from model.qwen2_config import Qwen2Config


class MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        # ff_dim_in = config.intermediate_size // 2
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        a1 = self.up_proj(hidden_states)
        a2 = self.gate_proj(hidden_states)
        intermediate_parallel = a1 * self.act_fn(a2)
        output = self.down_proj(intermediate_parallel)
        return output


class SelfAttention(nn.Module):
    def __init__(self, config: Qwen2Config, get_cos_sin):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim,
            bias=True)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=True)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=True)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)

        cos_sin = get_cos_sin()
        self.cos = cos_sin[0]
        self.sin = cos_sin[1]

    def forward(self, hidden_states: torch.Tensor):
        assert len(hidden_states.size()) == 3
        bsz, q_len, hsz = hidden_states.size()
        assert hsz == self.hidden_size
        all_q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        all_k = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)\
            .repeat_interleave(self.num_key_value_groups, dim=2)\
            .permute(1, 0, 2, 3)
        all_v = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)\
            .repeat_interleave(self.num_key_value_groups, dim=2)\
            .permute(0, 2, 1, 3)

        cos, sin = self.cos[:q_len], self.sin[:q_len]
        all_q = apply_rotary(all_q, cos, sin).permute(1, 2, 0, 3)
        all_k = apply_rotary(all_k, cos, sin).permute(1, 2, 0, 3)

        out = nn.functional.scaled_dot_product_attention(
            query=all_q, key=all_k, value=all_v, is_causal=True)

        out = out.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)
        out = self.o_proj(out)
        return out


class Block(nn.Module):
    def __init__(
            self, config: Qwen2Config,
            get_cos_sin):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = SelfAttention(config, get_cos_sin=get_cos_sin)
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def name_mapping(param: str):
    return "model."+param


class Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.padding_idx = None
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        get_cos_sin = precompute_cos_sin(
            config.rope_theta,
            config.max_position_embeddings,
            config.hidden_size//config.num_attention_heads,
            config.device)
        self.layers = nn.ModuleList(
            [Block(config, get_cos_sin)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.embed_tokens(input_ids)
        layers_output = []
        for layer in self.layers:
            layers_output.append(hidden_states.detach())
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        layers_output.append(hidden_states.detach())
        return hidden_states, layers_output

    def load_weights_from_hf(self, ref_model, model_id):
        """
        :return:
        """
        # model_id = "Qwen/Qwen2-1.5B"
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
