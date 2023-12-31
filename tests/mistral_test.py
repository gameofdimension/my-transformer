import unittest

import torch
from transformers import AutoModelForCausalLM
# from transformers.models.llama.modeling_llama import LlamaRMSNorm

from model.mistral import Model
from model.mistral_config import MistralConfig


class ModelTest(unittest.TestCase):

    def test_modeling(self):
        device = 'cuda'
        ref_model_id = "mistralai/Mistral-7B-v0.1"
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id).to(device)

        config = MistralConfig()
        model = Model(config)
        model.load_weights_from_hf(ref_model_id)
        model = model.to(device)

        # input_ids = torch.LongTensor([[42, 2]]).to(device)
        input_ids = torch.randint(high=config.vocab_size, size=(5, 6000))
        out1 = ref_model(
            input_ids, output_hidden_states=True)
        out2, layer_output = model(input_ids)

        delta = torch.abs(torch.max(out1.hidden_states[-1][0] - out2))
        print(delta)
        # self.assertTrue(delta < 1e-3, f"fail at final output, delta {delta}")

        for i in range(config.num_hidden_layers):
            t1 = out1.hidden_states[i][0]
            print(t1.dtype)
            t2 = layer_output[i]
            delta = torch.abs(torch.max(t2 - t1))
            print(f"{i}", delta)
            # self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
