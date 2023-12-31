import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN

from model.common import RMSNorm
from model.mistral_config import MistralConfig


class MLP(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor):
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        down = gate*up
        out = self.down_proj(down)
        return out


def precompute_attn_mask(sliding_window, q_len, device):
    def make_attn_mask():
        tmp1 = 1 - torch.triu(
            torch.ones((q_len, q_len), dtype=torch.int, requires_grad=False),
            diagonal=(1-sliding_window))
        tmp2 = torch.triu(torch.ones(
            (q_len, q_len), dtype=torch.int, requires_grad=False), diagonal=1)
        tmp = tmp1 + tmp2
        attn_mask = torch.zeros((q_len, q_len), requires_grad=False)
        attn_mask.masked_fill_(tmp.bool(), -float('inf'))
        return attn_mask

    attn_mask = make_attn_mask().to(device)

    def get_attn_mask():
        return attn_mask

    return get_attn_mask


def precompute_cos_sin(rope_theta, n: int, d: int, device):
    assert d > 0 and d % 2 == 0

    base = torch.tensor(rope_theta)
    cos = torch.zeros(n, d, requires_grad=False)
    sin = torch.zeros(n, d, requires_grad=False)
    for i in range(n):
        for j in range(d // 2):
            theta = base ** (-2 * j / d)
            cos[i, j] = torch.cos(i * theta)
            cos[i, j + d // 2] = torch.cos(i * theta)
            sin[i, j] = -torch.sin(i * theta)
            sin[i, j + d // 2] = torch.sin(i * theta)

    cos = cos.to(device)
    sin = sin.to(device)

    def get_cos_sin():
        return cos, sin

    return get_cos_sin


def apply_rotary(vector: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    assert vector.dim() == 4
    assert cos.dim() == sin.dim() == 2
    assert cos.size(-1) == sin.size(-1) == vector.size(-1)
    assert cos.size(0) == sin.size(0) == vector.size(0)
    sl, bs, nh, d = vector.size()
    cos = cos.view(sl, 1, 1, -1)
    sin = sin.view(sl, 1, 1, -1)
    tmp = torch.cat([vector[..., d // 2:], vector[..., :d // 2]], dim=-1)
    return vector * cos + tmp * sin


class SelfAttention(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int,
                 get_cos_sin, get_attn_mask):
        super().__init__()
        self.layer_idx = layer_idx
        self.sliding_window = config.sliding_window

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

        attn_mask = get_attn_mask()
        self.attn_mask = attn_mask

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

        attn_mask = self.attn_mask[:q_len, :q_len]
        cos, sin = self.cos[:q_len], self.sin[:q_len]
        all_q = apply_rotary(all_q, cos, sin).permute(1, 2, 0, 3)
        all_k = apply_rotary(all_k, cos, sin).permute(1, 2, 0, 3)
        # all_v = apply_rotary(all_v, cos, sin).permute(1, 2, 0, 3)
        out = nn.functional.scaled_dot_product_attention(
            query=all_q, key=all_k, value=all_v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)
        out = self.o_proj(out)
        return out


class Block(nn.Module):
    def __init__(
            self, config: MistralConfig, layer_idx: int,
            get_cos_sin, get_attn_mask):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SelfAttention(
            config=config, layer_idx=layer_idx,
            get_cos_sin=get_cos_sin, get_attn_mask=get_attn_mask)

        self.mlp = MLP(config)
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
        hidden_states = self.mlp(hidden_states)
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
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        get_cos_sin = precompute_cos_sin(
            config.rope_theta,
            config.max_position_embeddings,
            config.hidden_size//config.num_attention_heads,
            config.device)
        get_attn_mask = precompute_attn_mask(
            config.sliding_window,
            config.max_position_embeddings,
            config.device)
        self.layers = nn.ModuleList(
            [Block(config, layer_idx, get_cos_sin, get_attn_mask)
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

    def load_weights_from_hf(self, model_id):
        """
        :return:
        """
        # model_id = 'mistralai/Mistral-7B-v0.1'
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)
