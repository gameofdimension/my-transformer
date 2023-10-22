import unittest

import numpy as np
import torch

from flash.incremental import incremental_softmax_weighted_sum, incremental_softmax_weighted_sum_2d
from flash.v1 import FlashAttentionV1
from flash.v2 import FlashAttentionV2


class FlashAttentionV1Test(unittest.TestCase):
    def test_incremental_softmax(self):
        score = np.random.randn(17, )
        value = np.random.randn(17, 5)
        gold = torch.softmax(torch.tensor(score), dim=-1) @ torch.tensor(value)
        assert gold.size() == (5,)

        inc = incremental_softmax_weighted_sum(score, value, 5)
        self.assertTrue(torch.allclose(gold, torch.tensor(inc)))

    def test_incremental_softmax_weighted_sum_2d(self):
        score = np.random.randn(8, 17)
        value = np.random.randn(17, 6)
        gold = torch.softmax(torch.tensor(score), dim=-1) @ torch.tensor(value)
        assert gold.size() == (8, 6)

        inc1, inc2 = incremental_softmax_weighted_sum_2d(score, value, 5)
        self.assertTrue(torch.allclose(gold, torch.tensor(inc1)))

        self.assertTrue(torch.allclose(gold, torch.tensor(inc2)))

    def test_incremental_softmax_weighted_sum_1(self):
        score = np.random.randn(8, 17)
        value = np.random.randn(17, 6)
        gold = torch.softmax(torch.tensor(score), dim=-1) @ torch.tensor(value)
        assert gold.size() == (8, 6)

        for i in range(score.shape[0]):
            inc = incremental_softmax_weighted_sum(score[i], value, 5)
            self.assertTrue(torch.allclose(gold[i], torch.tensor(inc)))

    def test_v1(self):
        fl = FlashAttentionV1()
        q = np.random.randn(730, 64)
        k = np.random.randn(730, 64)
        v = np.random.randn(730, 64)

        out = fl.forward(q, k, v)
        gold = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(q), torch.tensor(k), torch.tensor(v))
        self.assertTrue(torch.allclose(gold, torch.tensor(out), atol=1e-6))

    def test_v2(self):
        fl = FlashAttentionV2()
        q = np.random.randn(730, 64)
        k = np.random.randn(730, 64)
        v = np.random.randn(730, 64)

        out, denominator = fl.forward(q, k, v)
        gold = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(q), torch.tensor(k), torch.tensor(v))
        self.assertTrue(torch.allclose(gold, torch.tensor(out), atol=1e-6))
