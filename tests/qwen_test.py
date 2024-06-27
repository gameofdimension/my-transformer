import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from model.qwen import Model
from model.qwen_config import QwenConfig


class ModelTest(unittest.TestCase):

    def test_modeling(self):
        device = 'cpu'
        ref_model_id = "Qwen/Qwen-1_8B"
        ref_config = AutoConfig.from_pretrained(
            ref_model_id, trust_remote_code=True)
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            config=ref_config,
            trust_remote_code=True,
        ).to(device, dtype=torch.float32)
        ref_model.eval()

        config = QwenConfig(
            device=device,
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
            a = out1.hidden_states[i]
            b = layer_output[i]
            delta = torch.abs(a - b).max()
            self.assertTrue(delta < 1e-2, f"fail at layer {i}, delta {delta}")
