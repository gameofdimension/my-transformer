import unittest

import torch

from model.llama import RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class ModelTest(unittest.TestCase):
    def test_rmsnorm(self):
        h = torch.normal(mean=0, std=1, size=(2, 3, 4, 10))

        norm = RMSNorm(hidden_size=10, eps=1e-5)
        out = norm(h)

        llnorm = LlamaRMSNorm(hidden_size=10, eps=1e-5)
        gold = llnorm(h)

        self.assertTrue(torch.max(torch.abs(out - gold)) < 1e-3)
