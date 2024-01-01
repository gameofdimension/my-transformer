import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN

from model.common import RMSNorm, precompute_cos_sin
from model.mistral import apply_rotary
from model.mixtral_config import MixtralConfig


class Top2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, routing_weights):
        current_hidden_states = self.act_fn(
            self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states


class Moe(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        assert self.top_k == 2

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Top2MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        logits = self.gate(hidden_states)
        probs = nn.functional.softmax(logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            probs, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        assert hidden_states.size(0) == routing_weights.size(
            0) == selected_experts.size(0)
        out = torch.zeros_like(hidden_states)
        for i in range(hidden_states.size(0)):
            for weight, idx in zip(routing_weights[i], selected_experts[i]):
                expert = self.experts[idx]
                out[i] += expert(hidden_states[i], weight)
        return out.reshape(bsz, seq_len, hidden_dim)


class SelfAttention(nn.Module):
    def __init__(self, config: MixtralConfig, layer_idx: int,
                 get_cos_sin):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=False)
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
            self, config: MixtralConfig, layer_idx: int,
            get_cos_sin):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SelfAttention(
            config=config, layer_idx=layer_idx,
            get_cos_sin=get_cos_sin)
        self.block_sparse_moe = Moe(config)

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def name_mapping(param: str):
    out = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
    }
    if param in out:
        return out[param]
    return "model."+param


class Model(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        get_cos_sin = precompute_cos_sin(
            config.rope_theta,
            config.max_position_embeddings,
            config.hidden_size//config.num_attention_heads,
            config.device)
        self.layers = nn.ModuleList(
            [Block(config, layer_idx, get_cos_sin)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.embed_tokens(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.final_norm(hidden_states), layers_output

    def load_weights_from_hf(self, ref_model, model_id):
        """
        :return:
        """
        # model_id = 'mistralai/Mixtral-8x7B-v0.1'
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
