import unittest

import torch
from transformers import AutoModel, AutoConfig

from model.chatglm import Model
from model.chatglm_config import ChatGLMConfig


class ModelTest(unittest.TestCase):

    def test_modeling_cpu(self):
        ref_model_id = "felixdae/chatglm-6b"
        config = AutoConfig.from_pretrained(ref_model_id, trust_remote_code=True)
        ref_model = AutoModel.from_pretrained(ref_model_id, config=config, trust_remote_code=True)
        ref_model = ref_model.float()

        config = ChatGLMConfig(num_layers=2)
        model = Model(config)
        model.load_weights_from_hf(ref_model_id)

        input_ids, position_ids = torch.LongTensor([42, 130001, 130004]), torch.LongTensor([[0, 1, 1], [0, 0, 1]])
        out1 = ref_model(input_ids.unsqueeze(0), position_ids.unsqueeze(0), output_hidden_states=True)
        out2, layer_output = model(input_ids, position_ids)

        delta = torch.abs(torch.max(out1.hidden_states[-1].transpose(0, 1)[0] - out2))
        self.assertTrue(delta < 1e-3, f"fail at final output, delta {delta}")

        for i in range(config.num_layers):
            t1 = out1.hidden_states[i].transpose(0, 1)[0]
            t2 = layer_output[i]
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")

    def test_modeling_gpu(self):
        ref_model_id = "felixdae/chatglm-6b"
        ref_model = AutoModel.from_pretrained(ref_model_id, trust_remote_code=True)
        ref_model = ref_model.half().cuda()

        config = ChatGLMConfig(num_layers=2)
        model = Model(config)
        model.load_weights_from_hf(ref_model_id)

        input_ids, position_ids = torch.LongTensor([42, 130001, 130004]), torch.LongTensor([[0, 1, 1], [0, 0, 1]])
        out1 = ref_model(input_ids.unsqueeze(0).cuda(), position_ids.unsqueeze(0).cuda(), output_hidden_states=True)
        out2, layer_output = model(input_ids, position_ids)

        delta = torch.abs(torch.max(out1.hidden_states[-1].cpu().transpose(0, 1)[0] - out2))
        self.assertTrue(delta < 1e-3, f"fail at final output, delta {delta}")

        for i in range(config.num_layers):
            t1 = out1.hidden_states[i].transpose(0, 1)[0]
            t2 = layer_output[i]
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
