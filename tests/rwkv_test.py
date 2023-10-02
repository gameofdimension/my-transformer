import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from model.rwkv import Model
from model.rwkv_config import RwkvConfig


class ModelTest(unittest.TestCase):
    def test_modeling(self):
        ref_model_id = "RWKV/rwkv-4-169m-pile"
        ref_config = AutoConfig.from_pretrained(ref_model_id)
        # what is the purpose of rescale_every?
        ref_config.rescale_every = 0

        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, config=ref_config)
        input_ids = torch.LongTensor([[2, 5, 13], [3, 4, 168]])
        ref_output = ref_model(input_ids, output_hidden_states=True)

        ref_hidden_states = ref_output.hidden_states

        model = Model(RwkvConfig())
        model.load_weights_from_hf(ref_model_id)
        last, hidden_states = model(input_ids)

        self.assertEqual(len(ref_hidden_states), len(hidden_states))
        for a, b in zip(ref_hidden_states, hidden_states):
            self.assertEqual(a.size(), b.size())
            self.assertTrue(torch.allclose(a, b, atol=1e-3))
