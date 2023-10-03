import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from model.rwkv import Model
from model.rwkv_config import RwkvConfig


class ModelTest(unittest.TestCase):
    def test_modeling_train(self):
        ref_model_id = "RWKV/rwkv-4-169m-pile"
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id)
        ref_model.train()
        input_ids = torch.LongTensor([[2, 5, 13], [3, 4, 168]])
        ref_output = ref_model(input_ids, output_hidden_states=True)

        ref_hidden_states = ref_output.hidden_states

        model = Model(RwkvConfig())
        model.load_weights_from_hf(ref_model_id)
        model.train()
        last, hidden_states = model(input_ids)

        self.assertEqual(len(ref_hidden_states), len(hidden_states))
        for a, b in zip(ref_hidden_states, hidden_states):
            self.assertEqual(a.size(), b.size())
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_modeling_eval(self):
        ref_model_id = "RWKV/rwkv-4-169m-pile"

        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id)
        ref_model.eval()
        input_ids = torch.LongTensor([[2, 5, 13], [3, 4, 168]])
        ref_output = ref_model(input_ids, output_hidden_states=True)

        ref_hidden_states = ref_output.hidden_states

        model = Model(RwkvConfig())
        model.load_weights_from_hf(ref_model_id)
        model.eval()
        last, hidden_states = model(input_ids)

        self.assertEqual(len(ref_hidden_states), len(hidden_states))
        for a, b in zip(ref_hidden_states, hidden_states):
            self.assertEqual(a.size(), b.size())
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
