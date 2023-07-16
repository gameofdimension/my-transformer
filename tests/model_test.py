import unittest

import numpy as np
import torch

from model.one_piece import weighted_sum


class ModelTest(unittest.TestCase):
    def softmax(self, scores):
        scores = [v.item() for v in scores]
        maxv = max(scores)
        scores = [v - maxv for v in scores]
        exp = [np.exp(v) for v in scores]
        z = sum(exp)
        return [v / z for v in exp]

    def my_sum(self, scores, values):
        probs = self.softmax(scores)
        assert len(probs) == len(values)
        tmp = torch.zeros_like(values[0])
        for p, v in zip(probs, values):
            tmp += p * v
        return tmp

    def test_weighted_sum(self):
        scores = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        values = [
            torch.Tensor([1, 2, 3, 4]),
            torch.Tensor([4, 5, 6, 7]),
            torch.Tensor([7, 2, 3, 1]),
        ]
        val = weighted_sum(scores, values)
        my_val = self.my_sum(scores, values)

        self.assertEqual(val.shape, my_val.shape)
        for i in range(val.shape[0]):
            self.assertAlmostEqual(val[i].item(), my_val[i].item(), delta=1e-6)
