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

        block_kv = math.ceil(self.sram_size / (4 * d))
        block_q = min(block_kv, d)

        step_kv = math.ceil(n / block_kv)
        step_q = math.ceil(n / block_q)

        numerator = torch.zeros((n, d))
        denominator = torch.zeros((n, 1))
        expo_max = torch.full((n, 1), -float('inf'))

        def key_block(idx: int):
            assert 0 <= idx < step_kv
            return key[idx * block_kv:(idx + 1) * block_kv, :]

        def value_block(idx: int):
            assert 0 <= idx < step_kv
            return value[idx * block_kv:(idx + 1) * block_kv, :]

        def query_block(idx: int):
            assert 0 <= idx < step_q
            return query[idx * block_q:(idx + 1) * block_q, :]

        def get_numerator_block(idx: int):
            assert 0 <= idx < step_q
            return numerator[idx * block_q:(idx + 1) * block_q, :]

        def set_numerator_block(idx: int, val: torch.Tensor):
            assert 0 <= idx < step_q
            numerator[idx * block_q:(idx + 1) * block_q, :] = val

        def get_denominator_block(idx: int):
            assert 0 <= idx < step_q
            return denominator[idx * block_q:(idx + 1) * block_q, :]

        def set_denominator_block(idx: int, val: torch.Tensor):
            assert 0 <= idx < step_q
            denominator[idx * block_q:(idx + 1) * block_q, :] = val

        def get_emax_block(idx: int):
            assert 0 <= idx < step_q
            return expo_max[idx * block_q:(idx + 1) * block_q, :]

        def set_emax_block(idx: int, val: torch.Tensor):
            assert 0 <= idx < step_q
            expo_max[idx * block_q:(idx + 1) * block_q, :] = val

        for j in range(step_kv):
            k, v = key_block(j), value_block(j)
            for i in range(step_q):
                q = query_block(i)
                s = q @ k.T / math.sqrt(d)

                ni, di, emi = get_numerator_block(i), get_denominator_block(i), get_emax_block(i)
                emb = torch.max(s, dim=1, keepdim=True).values
                pb = torch.exp(s - emb)
                db = torch.sum(pb, dim=1, keepdim=True)

                em_tmp = torch.maximum(emb, emi)
                d_tmp = torch.exp(emi - em_tmp) * di + torch.exp(emb - em_tmp) * db
                n_tmp = torch.exp(emi - em_tmp) * ni + torch.exp(emb - em_tmp) * (pb @ v)

                set_numerator_block(i, n_tmp)
                set_denominator_block(i, d_tmp)
                set_emax_block(i, em_tmp)

        return numerator / denominator


