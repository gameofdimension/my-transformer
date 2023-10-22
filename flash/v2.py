import math

import numpy as np


class FlashAttentionV2:
    def __init__(self, sram_size: int = 100_000):
        super().__init__()
        self.sram_size = sram_size

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        assert query.ndim == 2
        assert query.shape == key.shape == value.shape
        n, d = query.shape

        block_kv = math.ceil(self.sram_size / (4 * d))
        block_q = min(block_kv, d)

        step_kv = math.ceil(n / block_kv)
        step_q = math.ceil(n / block_q)

        numerator = np.zeros((n, d))
        denominator = np.zeros((n, 1))
        expo_max = np.full((n, 1), -float('inf'))

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

        def set_numerator_block(idx: int, val: np.ndarray):
            assert 0 <= idx < step_q
            numerator[idx * block_q:(idx + 1) * block_q, :] = val

        def get_denominator_block(idx: int):
            assert 0 <= idx < step_q
            return denominator[idx * block_q:(idx + 1) * block_q, :]

        def set_denominator_block(idx: int, val: np.ndarray):
            assert 0 <= idx < step_q
            denominator[idx * block_q:(idx + 1) * block_q, :] = val

        def get_emax_block(idx: int):
            assert 0 <= idx < step_q
            return expo_max[idx * block_q:(idx + 1) * block_q, :]

        def set_emax_block(idx: int, val: np.ndarray):
            assert 0 <= idx < step_q
            expo_max[idx * block_q:(idx + 1) * block_q, :] = val

        for i in range(step_q):
            q = query_block(i)
            ni, di, emi = get_numerator_block(i), get_denominator_block(i), get_emax_block(i)
            for j in range(step_kv):
                k, v = key_block(j), value_block(j)
                s = q @ k.T / math.sqrt(d)

                emb = s.max(axis=1, keepdims=True)
                pb = np.exp(s - emb)
                db = np.sum(pb, axis=1, keepdims=True)

                em_tmp = np.maximum(emb, emi)
                di = np.exp(emi - em_tmp) * di + np.exp(emb - em_tmp) * db
                ni = np.exp(emi - em_tmp) * ni + np.exp(emb - em_tmp) * (pb @ v)
                emi = em_tmp

            set_numerator_block(i, ni)
            set_denominator_block(i, di)
            set_emax_block(i, emi)

        return numerator / denominator, denominator
