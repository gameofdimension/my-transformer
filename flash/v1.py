import math

import torch
from torch import nn


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
                emb = s.amax(dim=1, keepdim=True)
                pb = torch.exp(s - emb)
                db = torch.sum(pb, dim=1, keepdim=True)

                em_tmp = torch.maximum(emb, emi)
                d_tmp = torch.exp(emi - em_tmp) * di + torch.exp(emb - em_tmp) * db
                n_tmp = torch.exp(emi - em_tmp) * ni + torch.exp(emb - em_tmp) * (pb @ v)

                set_numerator_block(i, n_tmp)
                set_denominator_block(i, d_tmp)
                set_emax_block(i, em_tmp)

        return numerator / denominator
