import math

import torch
from torch import nn


def incremental_softmax_weighted_sum(score: torch.Tensor, value: torch.Tensor, block_size: int):
    assert len(score.size()) == 1
    assert score.size(0) == value.size(0)

    def compute_one_block(vec: torch.Tensor, val: torch.Tensor):
        assert 0 < vec.size(0) <= block_size
        assert vec.size(0) == val.size(0)

        maxv = torch.max(vec)
        vec = vec - maxv
        exp_val = torch.exp(vec)
        total = torch.sum(exp_val)
        return maxv, exp_val @ val, total

    def merge_two_block(
            v1: torch.Tensor, l1: torch.Tensor, m1: torch.Tensor,
            v2: torch.Tensor, l2: torch.Tensor, m2: torch.Tensor):
        maxv = torch.max(m1, m2)
        total = l1 * torch.exp(m1 - maxv) + l2 * torch.exp(m2 - maxv)

        v1 = torch.exp(m1 - maxv) * v1
        v2 = torch.exp(m2 - maxv) * v2
        return maxv, v1 + v2, total

    if score.size(0) <= block_size:
        _, e, t = compute_one_block(score, value)
        return e / t

    m, e, t = compute_one_block(score[0:block_size], value[0:block_size, :])
    i = block_size

    while True:
        if i + block_size >= score.size(0):
            nm, ne, nt = compute_one_block(score[i:], value[i:, :])
            _, e, t = merge_two_block(e, t, m, ne, nt, nm)
            return e / t

        nm, ne, nt = compute_one_block(score[i:i + block_size], value[i:i + block_size, :])
        m, e, t = merge_two_block(e, t, m, ne, nt, nm)
        i += block_size


def incremental_softmax_weighted_sum_2d(score: torch.Tensor, value: torch.Tensor, block_size: int):
    """

    :param score: [r, seq length]
    :param value: [seq length, d]
    :param block_size:
    :return:
    """
    assert len(score.size()) == len(value.size()) == 2
    assert score.size(1) == value.size(0)

    def compute_one_block(vec: torch.Tensor, val: torch.Tensor):
        assert 0 < vec.size(1) <= block_size
        assert vec.size(1) == val.size(0)
        maxv = torch.max(vec, dim=1, keepdim=True).values

        vec = vec - maxv
        exp_val = torch.exp(vec)
        total = torch.sum(exp_val, dim=1, keepdim=True)
        return maxv, exp_val @ val, total

    def merge_two_block(
            v1: torch.Tensor, l1: torch.Tensor, m1: torch.Tensor,
            v2: torch.Tensor, l2: torch.Tensor, m2: torch.Tensor):
        maxv = torch.maximum(m1, m2)
        total = l1 * torch.exp(m1 - maxv) + l2 * torch.exp(m2 - maxv)

        v1 = torch.exp(m1 - maxv) * v1
        v2 = torch.exp(m2 - maxv) * v2
        return maxv, v1 + v2, total

    m = torch.full((score.size(0), 1), -float('inf'))
    e = torch.zeros((1, value.size(1)))
    t = torch.zeros((score.size(0), 1))

    i = 0
    while True:
        if i + block_size >= score.size(1):
            nm, ne, nt = compute_one_block(score[:, i:], value[i:, :])
            _, e, t = merge_two_block(e, t, m, ne, nt, nm)
            return e / t

        nm, ne, nt = compute_one_block(score[:, i:i + block_size], value[i:i + block_size, :])
        m, e, t = merge_two_block(e, t, m, ne, nt, nm)
        i += block_size


class FlashAttentionV1(nn.Module):
    def __init__(self, sram_size: int = 100_000):
        super().__init__()
        self.sram_size = sram_size

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        assert len(query.size()) == 2
        assert query.size() == key.size() == value.size()
        n, d = query.size()

        bc = math.ceil(self.sram_size / (4 * d))
        br = min(bc, d)
        tc = math.ceil(n / bc)
        tr = math.ceil(n / br)

        big_o = torch.zeros((n, d))
        little_l = torch.zeros((n,))
        little_m = torch.full((n,), -float('inf'))

        def key_block(idx: int):
            assert 0 <= idx < tc
            return key[idx * bc:(idx + 1) * bc, :]

        def value_block(idx: int):
            assert 0 <= idx < tc
            return value[idx * bc:(idx + 1) * bc, :]

        def query_block(idx: int):
            assert 0 <= idx < tr
            return query[idx * br:(idx + 1) * br, :]

        def o_block(idx: int):
            assert 0 <= idx < tr
            return big_o[idx * br:(idx + 1) * br, :]

        def l_block(idx: int):
            assert 0 <= idx < tr
            return little_l[idx * br:(idx + 1) * br]

        def m_block(idx: int):
            assert 0 <= idx < tr
            return little_m[idx * br:(idx + 1) * br]

        for j in range(tc):
            k, v = key_block(j), value_block(j)
            for i in range(tr):
                q, o, li, mi = query_block(i), o_block(i), l_block(i), m_block(i)
                s = q @ k.T

                mb = torch.max(s, dim=1, keepdim=True).values
                pb = torch.exp(s - mb)
                lb = torch.sum(pb, dim=1, keepdim=True)

                mtmp = torch.maximum(mb, mi)
                ltmp = torch.exp(mi - mtmp) * li + torch.exp(mb - mtmp) * lb
