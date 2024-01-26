import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from model.deepseek import Model
from model.deepseek_config import DeepseekConfig


class ModelTest(unittest.TestCase):

    def test_modeling(self):
        # torch.manual_seed(42)
        device = 'cpu'
        ref_model_id = "deepseek-ai/deepseek-moe-16b-chat"
        ref_config = AutoConfig.from_pretrained(
            ref_model_id, trust_remote_code=True)
        ref_config.torch_dtype = "float32"
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            config=ref_config,
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()

        config = DeepseekConfig(
            torch_dtype='float32',
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
            t1 = out1.hidden_states[i]
            t2 = layer_output[i]
            assert t1.size() == t2.size()
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
