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

        out, logsumexp = fl.forward(q, k, v)
        gold = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(q), torch.tensor(k), torch.tensor(v))
        self.assertTrue(torch.allclose(gold, torch.tensor(out), atol=1e-6))

    def test_dsoftmax(self):
        score = np.random.randn(6, 8)
        exp_score = np.exp(score - np.max(score, axis=1, keepdims=True))
        rowsum = np.sum(exp_score, axis=1, keepdims=True)
        prob = exp_score / rowsum
        dprob = np.random.randn(6, 8)

        gold_score = torch.tensor(score, requires_grad=True)
        gold = torch.nn.functional.softmax(gold_score, dim=1)
        self.assertTrue(torch.allclose(gold, torch.tensor(prob)))

        dscore = dsoftmax(dprob, prob)
        gold.backward(torch.tensor(dprob))
        gold_dscore = gold_score.grad

        torch.allclose(gold_dscore, torch.tensor(dscore), atol=1e-6)

    def test_v2_backward(self):
        fl = FlashAttentionV2()
        q = np.random.randn(730, 64)
        k = np.random.randn(730, 64)
        v = np.random.randn(730, 64)

        out, logsumexp = fl.forward(q, k, v)
        dout = np.random.randn(730, 64)
        dq, dk, dv = fl.backward(q, k, v, out, dout, logsumexp)

        tq = torch.tensor(q, requires_grad=True)
        tk = torch.tensor(k, requires_grad=True)
        tv = torch.tensor(v, requires_grad=True)
        gold = torch.nn.functional.scaled_dot_product_attention(tq, tk, tv)
        self.assertTrue(torch.allclose(gold, torch.tensor(out), atol=1e-6))

        tdout = torch.tensor(dout)
        gold.backward(tdout)

        self.assertTrue(torch.allclose(tq.grad, torch.tensor(dq)))
        self.assertTrue(torch.allclose(tk.grad, torch.tensor(dk)))
        self.assertTrue(torch.allclose(tv.grad, torch.tensor(dv)))
