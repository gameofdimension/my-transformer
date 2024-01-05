import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from model.phi2 import Model
from model.phi2_config import Phi2Config


class ModelTest(unittest.TestCase):

    def test_modeling(self):
        # torch.manual_seed(42)
        device = 'cpu'
        ref_model_id = "microsoft/phi-2"
        ref_config = AutoConfig.from_pretrained(
            ref_model_id, trust_remote_code=True)
        ref_config.torch_dtype = "float32"
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            config=ref_config,
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()

        config = Phi2Config(
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

        delta = torch.abs(torch.max(out1.logits - out2))
        self.assertTrue(delta < 1e-2, f"logits delta {delta}")
