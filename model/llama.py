import math
from typing import List

import torch
from torch import nn
from torch.nn.functional import softmax

from model.llama_config import LlamaConfig
from transformers.activations import get_activation


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        val = hidden_states * hidden_states
        norm = torch.sqrt(torch.mean(val, dim=-1, keepdim=True) + self.eps)
        return hidden_states / norm * self.weight


class Rotary:
    def __init__(self, d: int):
        assert d % 2 == 0
        self.d = d
        self.matrix_lst = []

    def _pad(self, target: int):
        base = 10000
        for m in range(len(self.matrix_lst) + 1, target + 1):
            matrix = torch.zeros(size=(self.d, self.d))

            for j in range(self.d // 2):
                theta = base ** (-2 * (j - 1) / self.d)
                matrix[2 * j, 2 * j] = math.cos(m * theta)
                matrix[2 * j, 2 * j + 1] = -math.sin(m * theta)
                matrix[2 * j + 1, 2 * j + 1] = math.cos(m * theta)
                matrix[2 * j + 1, 2 * j - 1] = math.sin(m * theta)
            assert m == len(self.matrix_lst) + 1
            self.matrix_lst.append(matrix)

    def apply(self, m: int, vec: torch.Tensor):
        assert m >= 1
        assert vec.size(-1) == self.d
        if m > len(self.matrix_lst):
            self._pad(m)
        matrix = self.matrix_lst[m - 1]
        return matrix @ vec


class Mlp(nn.Module):
    def __init__(self, config: LlamaConfig):
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


def weighted_sum(scores: List[torch.Tensor], values: List[torch.Tensor]) -> torch.Tensor:
    v = torch.stack(values)
    scores = torch.Tensor(scores)
    probs = softmax(scores, dim=-1)
    return probs @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.q_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.k_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.v_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.o_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

        self.config = config
        self.rotary = Rotary(self.config.hidden_size // self.config.num_attention_heads)

    def forward(self, hidden_states: torch.Tensor):
        # [seq_length, hidden_size]
        assert len(hidden_states.size()) == 2
        all_q = self.q_proj(hidden_states)
        all_k = self.k_proj(hidden_states)
        all_v = self.v_proj(hidden_states)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        def get_q(idx: int, head: int):
            return self.rotary.apply(idx + 1, all_q[idx, head * head_dim:(head + 1) * head_dim])

        def get_k(idx: int, head: int):
            return self.rotary.apply(idx + 1, all_k[idx, head * head_dim:(head + 1) * head_dim])

        def get_v(idx: int, head: int):
            return all_v[idx, head * head_dim:(head + 1) * head_dim]

        seq_length = hidden_states.shape[0]
        scale = head_dim ** 0.5
        output = []
        for tk in range(seq_length):
            final_value = []
            for h in range(self.config.num_attention_heads):
                q = get_q(tk, h)
                s = []
                v = []
                for p in range(1 + tk):
                    s.append(torch.dot(q, get_k(p, h)) / scale)
                    v.append(get_v(p, h))
                final_value.append(weighted_sum(s, v))
            final_value = torch.stack(final_value).view(1, -1).squeeze()
            assert final_value.shape[0] == self.config.hidden_size
            output.append(final_value)
        output = torch.stack(output)
        return self.o_proj(output)


class Block(nn.Module):
    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class Model(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.rms = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor):
        hidden_state = self.word_embedding_table(input_ids)
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return self.rms(hidden_state)


if __name__ == '__main__':
    r = Rotary(10)
    print(r.apply(3, torch.normal(mean=0, std=1, size=(10,))))

    model = Model(LlamaConfig())
    print(model)
    out = model(torch.LongTensor([42]))
    print(out.size(), out)
