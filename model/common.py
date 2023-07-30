import math
from typing import Callable, List

import torch
from torch import nn
from torch.nn.functional import softmax


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        val = hidden_states * hidden_states
        norm = torch.sqrt(torch.mean(val, dim=-1, keepdim=True) + self.eps)
        return hidden_states / norm * self.weight


def weighted_sum(scores: List[torch.Tensor], values: List[torch.Tensor]) -> torch.Tensor:
    v = torch.stack(values)
    scores = torch.Tensor(scores)
    probs = softmax(scores, dim=-1)
    return probs @ v


def slope_factory(num_heads: int):
    if num_heads & (num_heads - 1) == 0:
        base = 2 ** (-8 / num_heads)
        table = [base ** (i + 1) for i in range(num_heads)]
    else:
        floor_nh = 2 ** math.floor(math.log2(num_heads))
        base1 = 2 ** (-8 / floor_nh)
        table1 = [base1 ** (i + 1) for i in range(floor_nh)]

        base2 = 2 ** (-8 / (2 * floor_nh))
        table2 = [base2 ** (i + 1) for i in range(2 * floor_nh)]

        table = table1 + table2[::2][:num_heads - floor_nh]

    assert len(table) == num_heads

    def get_m(h: int):
        return table[h]

    return get_m


def attention_func(
        seq_length: int, num_attention_heads: int, hidden_size: int,
        get_q: Callable, get_k: Callable, get_v: Callable, alibi_get_m: Callable = None):
    assert hidden_size % num_attention_heads == 0
    head_dim = hidden_size // num_attention_heads
    scale = head_dim ** 0.5
    output = []
    for tk in range(seq_length):
        final_value = []
        for h in range(num_attention_heads):
            q = get_q(tk, h)
            scores = []
            v = []
            for p in range(1 + tk):
                if alibi_get_m is None:
                    score = torch.dot(q, get_k(p, h)) / scale
                else:
                    score = torch.dot(q, get_k(p, h)) / scale + (p - tk) * alibi_get_m(h)
                scores.append(score)
                v.append(get_v(p, h))
            final_value.append(weighted_sum(scores, v))
        final_value = torch.stack(final_value).view(1, -1).squeeze()
        assert final_value.shape[0] == hidden_size
        output.append(final_value)
    output = torch.stack(output)
    return output


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
                theta = base ** (-2 * j / self.d)
                matrix[2 * j, 2 * j] = math.cos(m * theta)
                matrix[2 * j, 2 * j + 1] = -math.sin(m * theta)
                matrix[2 * j + 1, 2 * j + 1] = math.cos(m * theta)
                matrix[2 * j + 1, 2 * j] = math.sin(m * theta)
            assert m == len(self.matrix_lst) + 1
            self.matrix_lst.append(matrix)

    def apply(self, m: int, vec: torch.Tensor):
        assert m >= 1
        assert vec.size(-1) == self.d
        if m > len(self.matrix_lst):
            self._pad(m)
        matrix = self.matrix_lst[m - 1]
        return matrix @ vec
