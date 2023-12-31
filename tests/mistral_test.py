import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from model.mistral import Model
from model.mistral_config import MistralConfig


class ModelTest(unittest.TestCase):

    def test_modeling(self):
        device = 'cuda'
        ref_model_id = "mistralai/Mistral-7B-v0.1"
        ref_config = AutoConfig.from_pretrained(ref_model_id)
        ref_config.num_hidden_layers = 8
        ref_config.sliding_window = 16
        ref_config.max_position_embeddings = 2048
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            config=ref_config,
        ).to(device)
        ref_model.eval()

        config = MistralConfig(
            num_hidden_layers=8,
            sliding_window=16,
            max_position_embeddings=2048
        )
        model = Model(config)
        model.load_weights_from_hf(ref_model, None)
        model = model.to(device)
        model.eval()

        input_ids = torch.randint(
            high=config.vocab_size, size=(5, 64),
            device=device, requires_grad=False)

        with torch.no_grad():
            out1 = ref_model(
                input_ids, output_hidden_states=True)
            out2, layer_output = model(input_ids)

        for i in range(config.num_hidden_layers):
            t1 = out1.hidden_states[i]
            t2 = layer_output[i]
            assert t1.size() == t2.size()
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
