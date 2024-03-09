import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from model.gemma import Model
from model.gemma_config import GemmaConfig


class ModelTest(unittest.TestCase):

    def test_2b_modeling(self):
        device = 'cpu'
        # config for 7b model
        # config = GemmaConfig(
        #     device=device,
        #     max_position_embeddings=1000,
        # )
        # ref_model_id = "google/gemma-7b"

        # config for 2b model
        config = GemmaConfig(
            device=device,
            hidden_size=2048,
            intermediate_size=16384,
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            max_position_embeddings=1000,
        )
        ref_model_id = "google/gemma-2b"
        self.check_modeling(device, ref_model_id, config)

    def check_modeling(self, device, ref_model_id, config: GemmaConfig):
        # torch.manual_seed(42)
        ref_config = AutoConfig.from_pretrained(
            ref_model_id, trust_remote_code=True)
        ref_config.torch_dtype = "float32"
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            config=ref_config,
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()

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
            self.assertTrue(delta < 1e-2, f"fail at layer {i}, delta {delta}")
