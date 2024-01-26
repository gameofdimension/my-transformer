import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN

from model.common import RMSNorm, apply_rotary, precompute_cos_sin
from model.deepseek_config import DeepseekConfig


class SelfAttention(nn.Module):
    def __init__(self, config: DeepseekConfig, layer_idx: int,
                 get_cos_sin):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        assert self.num_key_value_heads == self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=config.attention_bias)
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


class MLP(nn.Module):
    def __init__(
            self, hidden_act: str, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor):
        out = self.down_proj(self.act_fn(
            self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return out


class Gate(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(
            config.hidden_size, config.n_routed_experts, bias=False)
        assert config.scoring_func == 'softmax'
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor):
        logits = self.linear(hidden_states)
        probs = nn.functional.softmax(logits, dim=-1)

        routing_weights, selected_experts = torch.topk(
            probs, self.top_k, dim=-1)

        if self.config.norm_topk_prob:
            routing_weights /= routing_weights.sum(
                dim=-1, keepdim=True) + 1e-20

        return routing_weights, selected_experts


class MoE(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([MLP(
            hidden_act=config.hidden_act,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size)
            for i in range(config.n_routed_experts)])
        self.gate = Gate(config)
        assert config.n_shared_experts is not None
        intermediate_size = config.moe_intermediate_size * \
            config.n_shared_experts
        self.shared_experts = MLP(
            hidden_act=config.hidden_act,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        routing_weights, selected_experts = self.gate(hidden_states)

        assert hidden_states.size(0) == routing_weights.size(
            0) == selected_experts.size(0)
        out = torch.zeros_like(hidden_states)
        for i in range(hidden_states.size(0)):
            out[i] += self.shared_experts(hidden_states[i])
            for weight, idx in zip(routing_weights[i], selected_experts[i]):
                expert = self.experts[idx]
                out[i] += weight*expert(hidden_states[i])
        return out.reshape(bsz, seq_len, hidden_dim)


class Block(nn.Module):
    def __init__(
            self, config: DeepseekConfig, layer_idx: int,
            get_cos_sin):
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SelfAttention(
            config=config, layer_idx=layer_idx,
            get_cos_sin=get_cos_sin)
        assert config.n_routed_experts is not None
        if layer_idx >= config.first_k_dense_replace and \
                layer_idx % config.moe_layer_freq == 0:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(
                hidden_act=config.hidden_act, hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size)

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
    out = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
    }
    if param in out:
        return out[param]
    name = "model."+param
    if name.endswith('gate.linear.weight'):
        return name.replace('gate.linear.weight', 'gate.weight')
    return name


class Model(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx)
        get_cos_sin = precompute_cos_sin(
            config.rope_theta,
            config.max_position_embeddings,
            config.hidden_size//config.num_attention_heads,
            config.device)
        self.layers = nn.ModuleList(
            [Block(config, layer_idx, get_cos_sin)
             for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.embed_tokens(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.norm(hidden_states), layers_output

    def load_weights_from_hf(self, ref_model, model_id):
        """
        :return:
        """
        # model_id = 'deepseek-ai/deepseek-moe-16b-chat'
        if ref_model is None:
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)
