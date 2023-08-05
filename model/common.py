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


def glm_attention_func(
        seq_length: int, num_attention_heads: int, hidden_size: int,
        gmask_pos: int, get_q: Callable, get_k: Callable, get_v: Callable):
    """
    清华的 glm 架构下的 attention 计算，跟标准的 attention 差别比较大。差异体现在：
    1. 2d 位置编码，从而导致不同的向量旋转算法
    2. context 部分是可以双向注意力的，也是就是 gmask_pos 之前的位置（包含）
    """
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
            if tk <= gmask_pos:
                end = gmask_pos
            else:
                end = tk
            for p in range(end + 1):
                score = torch.dot(q, get_k(p, h)) / scale
                scores.append(score)
                v.append(get_v(p, h))
            ws = weighted_sum(scores, v)
            final_value.append(ws)
        final_value = torch.stack(final_value).view(1, -1).squeeze()
        assert final_value.shape[0] == hidden_size
        output.append(final_value)
    output = torch.stack(output)
    return output


class Rotary:
    def __init__(self, d: int, paper: bool = False):
        assert d % 2 == 0
        self.d = d
        self.matrix_lst = []
        self.paper = paper

    def _pad(self, target: int):
        base = 10000
        for m in range(len(self.matrix_lst), target + 1):
            matrix = torch.zeros(size=(self.d, self.d))

            for j in range(self.d // 2):
                theta = base ** (-2 * j / self.d)
                # 以下是论文实现
                if self.paper:
                    matrix[2 * j, 2 * j] = math.cos(m * theta)
                    matrix[2 * j, 2 * j + 1] = -math.sin(m * theta)
                    matrix[2 * j + 1, 2 * j + 1] = math.cos(m * theta)
                    matrix[2 * j + 1, 2 * j] = math.sin(m * theta)
                # 以下是 llama 实现
                else:
                    matrix[j, j] = math.cos(m * theta)
                    matrix[j, j + self.d // 2] = -math.sin(m * theta)
                    matrix[j + self.d // 2, j + self.d // 2] = math.cos(m * theta)
                    matrix[j + self.d // 2, j] = math.sin(m * theta)
            self.matrix_lst.append(matrix)

    def apply(self, m: int, vec: torch.Tensor):
        assert m >= 0
        assert vec.size(-1) == self.d
        if m >= len(self.matrix_lst):
            self._pad(m)
        matrix = self.matrix_lst[m]
        return matrix @ vec


def vectorized_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = False):
    # [b,s,h,d]
    assert len(query.size()) == 4
    assert len(key.size()) == 4
    assert len(value.size()) == 4
    b, s, h, d = query.size()

    query = query.transpose(1, 2).reshape(b * h, s, d)
    key = key.transpose(1, 2).reshape(b * h, s, d).transpose(1, 2)
    value = value.transpose(1, 2).reshape(b * h, s, d)

    base = torch.zeros((1, 1, 1))
    scores = torch.baddbmm(input=base, batch1=query, batch2=key, beta=0) / math.sqrt(d)
    assert scores.size() == (b * h, s, s)
    if is_causal:
        attention_mask = torch.ones(1, s, s).tril()
        attention_mask = attention_mask.masked_fill(attention_mask == 0.0, -float('inf'))
        attention_mask = attention_mask.masked_fill(attention_mask > 0.0, 0)
        scores += attention_mask
    probs = torch.softmax(scores, dim=-1)
    output = torch.baddbmm(input=base, batch1=probs, batch2=value, beta=0)

    output = output.reshape(b, h, s, d).transpose(1, 2)
    return output


if __name__ == '__main__':
    attn_mask = torch.ones(5, 5).tril(diagonal=0)
    print(attn_mask)
    attn_mask = attn_mask.masked_fill(attn_mask == 0.0, -float('inf'))
    attn_mask = attn_mask.masked_fill(attn_mask > 0.0, 0)
    print(attn_mask)
