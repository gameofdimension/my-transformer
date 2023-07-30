import torch
from torch import nn
from transformers import AutoModel

from model.chatglm_config import ChatGLMConfig
from model.common import Rotary, attention_func


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class Mlp(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(in_features=config.hidden_size, out_features=config.inner_hidden_size)
        self.dense_4h_to_h = nn.Linear(in_features=config.inner_hidden_size, out_features=config.hidden_size)
        self.activation_func = gelu

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, layer_id: int, config: ChatGLMConfig):
        super().__init__()
        self.query_key_value = nn.Linear(in_features=config.hidden_size, out_features=3 * config.hidden_size)
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        assert config.hidden_size % (2 * config.num_attention_heads) == 0
        self.rotary = Rotary(config.hidden_size // (2 * config.num_attention_heads))
        self.config = config

    def forward(self, hidden_states: torch.LongTensor, position_ids: torch.LongTensor):
        assert len(position_ids.size()) == 2 and position_ids.size(0) == 2
        all_qkv = self.query_key_value(hidden_states)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        def get_q(idx: int, head: int):
            base = head * (3 * head_dim)
            q1q2 = all_qkv[idx, base + 0 * head_dim:base + 1 * head_dim]
            p = position_ids[0][idx]
            b = position_ids[1][idx]
            q1 = self.rotary.apply(p + 1, q1q2[:head_dim // 2])
            q2 = self.rotary.apply(b + 1, q1q2[head_dim // 2:])
            return torch.concat([q1, q2])

        def get_k(idx: int, head: int):
            base = head * (3 * head_dim)
            k1k2 = all_qkv[idx, base + 1 * head_dim:base + 2 * head_dim]
            p = position_ids[0][idx]
            b = position_ids[1][idx]
            k1 = self.rotary.apply(p + 1, k1k2[:head_dim // 2])
            k2 = self.rotary.apply(b + 1, k1k2[head_dim // 2:])
            return torch.concat([k1, k2])

        def get_v(idx: int, head: int):
            base = head * (3 * head_dim)
            return all_qkv[idx, base + 2 * head_dim:base + 3 * head_dim]

        seq_length = hidden_states.shape[0]
        output = attention_func(
            seq_length=seq_length, num_attention_heads=self.config.num_attention_heads,
            hidden_size=self.config.hidden_size, get_q=get_q, get_k=get_k, get_v=get_v)
        return self.dense(output)


class Block(nn.Module):
    def __init__(self, layer_id: int, config: ChatGLMConfig):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.mha = MultiHeadAttention(layer_id=layer_id, config=config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = Mlp(config)

        self.alpha = (2 * config.num_layers) ** 0.5

    def forward(self, hidden_states: torch.LongTensor, position_ids: torch.LongTensor):
        attention_input = self.input_layernorm(hidden_states)
        attention_output = self.mha(attention_input, position_ids)
        hidden_states = self.alpha * attention_input + attention_output
        print("1111", hidden_states[:, :5])

        mlp_input = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.alpha * mlp_input + mlp_output
        print("2222", hidden_states[:, :5])
        return hidden_states


def name_mapping(param: str):
    if '.mha.' in param:
        param = param.replace('.mha.', '.attention.')
    return 'transformer.' + param


class Model(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([Block(layer_id=i, config=config) for i in range(config.num_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor):
        assert len(input_ids.size()) == 1
        hidden_states = self.word_embeddings(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
            layers_output.append(hidden_states.detach())
        return self.final_layernorm(hidden_states), layers_output

    def load_weights_from_hf(self, model_id):
        """
        :return:
        """
        # model_id = 'felixdae/chatglm-6b'
        ref_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)
