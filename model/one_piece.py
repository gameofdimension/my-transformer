import math
from dataclasses import asdict
from typing import List

import torch
from torch import nn
from torch.nn.functional import softmax
from transformers.activations import get_activation

from model.config import Gpt2Config


class Mlp(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_fn = get_activation(config.activation_function)

        self.up_proj = nn.Linear(in_features=config.n_embd, out_features=4 * config.n_embd)
        self.down_proj = nn.Linear(in_features=4 * config.n_embd, out_features=config.n_embd)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        return self.down_proj(self.action_fn(self.up_proj(x)))


def weighted_sum(scores: List[torch.Tensor], values: List[torch.Tensor]) -> torch.Tensor:
    v = torch.stack(values)
    probs = softmax(torch.Tensor(scores), dim=-1)
    return probs @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config.n_embd % config.n_head == 0
        dim = config.n_embd // config.n_head
        self.qkv_dim = 3 * dim
        self.qkv_proj = nn.Linear(config.n_embd, self.qkv_dim * config.n_head)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """

        all_head_qkv = self.qkv_proj(hidden_states)
        dim = self.qkv_dim // 3

        def get_q(idx: int, head: int):
            base = 3 * dim * head
            return all_head_qkv[idx, base:base + dim]

        def get_k(idx: int, head: int):
            base = 3 * dim * head
            return all_head_qkv[idx, base + dim:base + 2 * dim]

        def get_v(idx: int, head: int):
            base = 3 * dim * head
            return all_head_qkv[idx, base + 2 * dim:base + 3 * dim]

        seq_length = hidden_states.shape[0]
        scale = math.sqrt(self.config.n_embd)
        output = []
        for tk in range(seq_length):
            final_value = []
            for h in range(self.config.n_head):
                q = get_q(tk, h)
                s = []
                v = []
                for p in range(1 + tk):
                    s.append(torch.dot(q, get_k(p, h)) / scale)
                    v.append(get_v(p, h))
                final_value.append(weighted_sum(s, v))
            final_value = torch.stack(final_value).view(1, -1).squeeze()
            assert final_value.shape[0] == self.config.n_embd
            output.append(final_value)
        output = torch.stack(output)
        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ln1 = nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mha = MultiHeadAttention(config)

        self.ln2 = nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = Mlp(config)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        for f, g in zip([self.mha, self.mlp], [self.ln1, self.ln2]):
            fx = f(g(x))
            x = x + fx
        return x


class Model(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        """

        :param input_ids: [seq_length]
        :return:
        """
        assert len(input_ids.shape) == 1
        position_ids = torch.LongTensor([range(input_ids.shape[0])])

        word_embeddings = self.word_embedding_table(input_ids).squeeze()
        position_embeddings = self.position_embedding_table(position_ids).squeeze()

        hidden_states = word_embeddings + position_embeddings

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.ln(hidden_states)

    def load_weights_from_hf(self):
        pass


def main():
    # config = AutoConfig.from_pretrained('gpt2')
    # m = Model(config)

    # print(m(torch.LongTensor([3, 4, 5, 2])))

    # ln = nn.LayerNorm(normalized_shape=5)
    # x1 = torch.Tensor([1, 5, 2, 4, 3])
    # print(ln(x1))

    # input = torch.normal(0, 1, (5, config.n_embd))
    # mlp = Mlp(config)
    # print(mlp(input).shape)
    # attn = MultiHeadAttention(config)
    # print(attn(input))

    config = Gpt2Config()
    print(config, asdict(config))

    input_ids = torch.LongTensor([7, 5, 6, 1, 8, 9])
    model = Model(config)
    output = model(input_ids)
    print(output.shape)
    for params in model.state_dict():
        print(params)


if __name__ == '__main__':
    main()
