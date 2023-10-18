import unittest

import torch

from flash.v1 import incremental_softmax_weighted_sum, incremental_softmax_weighted_sum_2d


class AttentionTest(unittest.TestCase):
    def test_incremental_softmax(self):
        score = torch.randn((17,))
        value = torch.randn((17, 5))
        gold = torch.softmax(score, dim=-1) @ value
        assert gold.size() == (5,)

        inc = incremental_softmax_weighted_sum(score, value, 5)
        self.assertTrue(torch.allclose(gold, inc))

    def test_incremental_softmax_weighted_sum_2d(self):
        score = torch.randn((8, 17))
        value = torch.randn((17, 6))
        gold = torch.softmax(score, dim=-1) @ value
        assert gold.size() == (8, 6)

        inc = incremental_softmax_weighted_sum_2d(score, value, 5)
        self.assertTrue(torch.allclose(gold, inc))

    def test_incremental_softmax_weighted_sum_2d_1(self):
        score = torch.randn((8, 17))
        value = torch.randn((17, 6))
        gold = torch.softmax(score, dim=-1) @ value
        assert gold.size() == (8, 6)

        for i in range(score.size(0)):
            inc = incremental_softmax_weighted_sum(score[i], value, 5)
            self.assertTrue(torch.allclose(gold[i], inc))
