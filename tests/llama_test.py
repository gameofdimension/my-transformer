import unittest

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from model.llama import RMSNorm, Model
from model.llama_config import LlamaConfig


class ModelTest(unittest.TestCase):
    def test_rmsnorm(self):
        h = torch.normal(mean=0, std=1, size=(2, 3, 4, 10))

        norm = RMSNorm(hidden_size=10, eps=1e-5)
        out = norm(h)

        llnorm = LlamaRMSNorm(hidden_size=10, eps=1e-5)
        gold = llnorm(h)

        self.assertTrue(torch.max(torch.abs(out - gold)) < 1e-3)

    def test_modeling(self):
        ref_model_id = "felixdae/Llama-2-7b-hf"
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id)

        config = LlamaConfig(num_hidden_layers=2)
        model = Model(config)
        model.load_weights_from_hf(ref_model_id)

        out1 = ref_model(torch.LongTensor([[42, 2]]), output_hidden_states=True)
        out2, layer_output = model(torch.LongTensor([42, 2]))

        delta = torch.abs(torch.max(out1.hidden_states[-1][0] - out2))
        self.assertTrue(delta < 1e-3, f"fail at final output, delta {delta}")

        for i in range(config.num_hidden_layers):
            t1 = out1.hidden_states[i][0]
            t2 = layer_output[i]
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
