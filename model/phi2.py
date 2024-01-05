import torch
from torch import nn
from model.common import apply_rotary, precompute_cos_sin
from model.phi2_config import Phi2Config
from transformers.activations import ACT2FN
from transformers import AutoModelForCausalLM


class MLP(nn.Module):

    def __init__(self, config: Phi2Config):
        super().__init__()
        n_inner = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, config.n_embd)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor):
        out = self.fc2(self.act(self.fc1(hidden_states)))
        return out


class SelfAttention(nn.Module):

    def __init__(self, config: Phi2Config, block_idx: int, get_cos_sin):
        super().__init__()
        self.block_idx = block_idx

        self.head_dim = config.n_embd // config.n_head
        self.n_head = self.n_head_kv = config.n_head
        op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        self.Wqkv = nn.Linear(config.n_embd, op_size, bias=True)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_pdrop = config.attn_pdrop

        cos_sin = get_cos_sin()
        self.cos = cos_sin[0]
        self.sin = cos_sin[1]

    def forward(self, hidden_states: torch.Tensor):
        bsz, q_len, _, = hidden_states.size()
        qkv = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q = q.reshape(bsz, q_len, self.n_head, -1).permute(1, 0, 2, 3)
        k = k.reshape(bsz, q_len, self.n_head, -1).permute(1, 0, 2, 3)
        v = v.reshape(bsz, q_len, self.n_head, -1).permute(0, 2, 1, 3)
        cos, sin = self.cos[:q_len], self.sin[:q_len]
        q = apply_rotary(q, cos, sin).permute(1, 2, 0, 3)
        k = apply_rotary(k, cos, sin).permute(1, 2, 0, 3)

        out = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, dropout_p=self.attn_pdrop)
        out = out.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)
        out = self.out_proj(out)
        return out


class Block(nn.Module):

    def __init__(self, config: Phi2Config, block_idx: int, get_cos_sin):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx

        self.mixer = SelfAttention(config,
                                   block_idx=block_idx,
                                   get_cos_sin=get_cos_sin)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states

        hidden_states = self.ln(hidden_states)
        attn_output = self.resid_dropout(self.mixer(hidden_states))
        mlp_output = self.resid_dropout(self.mlp(hidden_states))
        return residual + attn_output + mlp_output


def name_mapping(param: str):
    out = {
        "ln.weight": "lm_head.ln.weight",
        "ln.bias": "lm_head.ln.bias",
        "wte.weight": "transformer.embd.wte.weight",
    }
    if param in out:
        return out[param]
    return "transformer."+param


class Model(nn.Module):

    def __init__(self, config: Phi2Config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        rope_theta = 10000.0
        get_cos_sin = precompute_cos_sin(rope_theta, config.n_positions,
                                         config.n_embd // config.n_head,
                                         config.device)
        self.h = nn.ModuleList([
            Block(config, block_idx=i, get_cos_sin=get_cos_sin)
            for i in range(config.n_layer)
        ])

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        layers_output = [hidden_states.detach()]
        for layer in self.h:
            hidden_states = layer(hidden_states, )
            layers_output.append(hidden_states.detach())
        return self.ln(hidden_states), hidden_states

    def load_weights_from_hf(self, ref_model, model_id):
        """
        :return:
        """
        # model_id = 'microsoft/phi-2'
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
