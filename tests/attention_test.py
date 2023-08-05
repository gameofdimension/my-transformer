import unittest

import torch

from model.common import vectorized_attention


class AttentionTest(unittest.TestCase):
    def test_bilateral(self):
        b, s, h, d = 5, 6, 7, 8
        query = torch.randn((b, s, h, d))
        key = torch.randn((b, s, h, d))
        value = torch.randn((b, s, h, d))

        gold = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        my = vectorized_attention(query, key, value)
        self.assertTrue(torch.allclose(gold, my, atol=1e-6))

    def test_unilateral(self):
        b, s, h, d = 5, 6, 7, 8
        query = torch.randn((b, s, h, d))
        key = torch.randn((b, s, h, d))
        value = torch.randn((b, s, h, d))

        gold = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            is_causal=True
        ).transpose(1, 2)
        my = vectorized_attention(query, key, value, is_causal=True)
        self.assertTrue(torch.allclose(gold, my, atol=1e-6))
