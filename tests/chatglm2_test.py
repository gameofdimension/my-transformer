import unittest

import torch
from transformers import AutoConfig, AutoModel

from model.chatglm2 import Model
from model.chatglm2_config import ChatGLM2Config


class ModelTest(unittest.TestCase):

    def test_modeling_cpu(self):
        ref_model_id = "felixdae/chatglm2-6b"
        config = AutoConfig.from_pretrained(ref_model_id, trust_remote_code=True)
        ref_model = AutoModel.from_pretrained(ref_model_id, config=config, trust_remote_code=True)
        ref_model = ref_model.float()

        config = ChatGLM2Config(num_layers=2)
        model = Model(config)
        model.load_weights_from_hf(ref_model_id)

        input_ids = torch.LongTensor([42, 24])
        out1 = ref_model(input_ids.unsqueeze(0), output_hidden_states=True)
        out2, layer_output = model(input_ids)

        delta = torch.abs(torch.max(out1.hidden_states[-1].transpose(0, 1)[0] - out2))
        # print(out1.hidden_states[-1].size(), out2.size())
        # self.assertTrue(delta < 1e-3, f"fail at final output, delta {delta}")

        print()
        for i in range(config.num_layers):
            t1 = out1.hidden_states[i].transpose(0, 1)[0]
            t2 = layer_output[i]
            print(t1[:, :5])
            print(t2[:, :5])
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
