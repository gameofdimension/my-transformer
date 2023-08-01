import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from model.chatglm2_config import ChatGLM2Config
from model.common import RMSNorm, attention_func, Rotary


class Mlp(nn.Module):
    def __init__(self, config: ChatGLM2Config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias=False)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        return self.dense_4h_to_h(hidden_states)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ChatGLM2Config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.kv_channels == config.hidden_size // config.num_attention_heads
        self.qkv_hidden_size = config.hidden_size + 2 * config.multi_query_group_num * config.kv_channels
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size, bias=True)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = Rotary(config.kv_channels // 2)
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        # [seq_length, hidden_size]
        assert len(hidden_states.size()) == 2
        all_qkv = self.query_key_value(hidden_states)

        head_num = self.config.num_attention_heads
        head_dim = self.config.kv_channels

        def get_q(idx: int, head: int):
            query = all_qkv[idx, head * head_dim:(head + 1) * head_dim]
            return torch.cat([self.rotary.apply(idx, query[:head_dim // 2]), query[head_dim // 2:]])

        def get_k(idx: int, head: int):
            base = head_num * head_dim
            key_group = all_qkv[idx, base:base + self.config.multi_query_group_num * head_dim]
            cursor = (head % self.config.multi_query_group_num) * head_dim
            key = key_group[cursor:cursor + head_dim]
            return torch.cat([self.rotary.apply(idx, key[:head_dim // 2]), key[head_dim // 2:]])

        def get_v(idx: int, head: int):
            base = head_num * head_dim + self.config.multi_query_group_num * head_dim
            value_group = all_qkv[idx, base:base + self.config.multi_query_group_num * head_dim]
            cursor = (head % self.config.multi_query_group_num) * head_dim
            value = value_group[cursor:cursor + head_dim]
            return value

        seq_length = hidden_states.shape[0]
        output = attention_func(
            seq_length=seq_length, num_attention_heads=self.config.num_attention_heads,
            hidden_size=self.config.hidden_size, get_q=get_q, get_k=get_k, get_v=get_v)
        return self.dense(output)


class Block(nn.Module):
    def __init__(self, config: ChatGLM2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = MultiHeadAttention(config=config)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = Mlp(config=config)

    def forward(self, hidden_states: torch.Tensor):
        x = self.input_layernorm(hidden_states)
        x = self.self_attention(x)
        hidden_states += x

        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        hidden_states += x
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: ChatGLM2Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=config.padded_vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([Block(config=config) for _ in range(config.num_layers)])
        self.final_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        assert len(input_ids.size()) == 1
        hidden_states = self.word_embeddings(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.final_layernorm(hidden_states), layers_output

    def load_weights_from_hf(self, model_id):
        """
        :return:
        """
        # model_id = 'felixdae/chatglm2-6b'
        ref_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)


def name_mapping(param: str):
    if 'word_embeddings.weight' in param:
        return 'transformer.embedding.word_embeddings.weight'
    if 'final_layernorm.weight' in param:
        return 'transformer.encoder.final_layernorm.weight'
    return 'transformer.encoder.' + param
