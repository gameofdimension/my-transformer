import math
import unittest

import torch
from transformers import AutoModelForCausalLM

from model.baichuan13b import Model
from model.baichuan13b_config import Baichuan13bConfig
from model.common import slope_factory


def _get_interleave(n):
    """
    https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/modeling_baichuan.py#L20
    :param n:
    :return:
    """

    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


class ModelTest(unittest.TestCase):

    def test_slopes(self):
        def _check(n: int):
            gold = _get_interleave(n)
            get_m = slope_factory(n)
            for i in range(n):
                self.assertAlmostEqual(gold[i], get_m(i), delta=1e-6, msg=f"at {i}, {gold[i]} vs {get_m(i)}")

        _check(4)
        _check(5)
        _check(8)
        _check(9)
        _check(233)
        _check(5000)

    def test_modeling(self):
        ref_model_id = "felixdae/Baichuan-13B-Chat"
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, trust_remote_code=True)

        config = Baichuan13bConfig(num_hidden_layers=2)
        model = Model(config)
        model.load_weights_from_hf(ref_model_id)

        out1 = ref_model(torch.LongTensor([[42, 2]]), output_hidden_states=True)
        out2, layer_output = model(torch.LongTensor([42, 2]))

        delta = torch.abs(torch.max(out1.hidden_states[-1][0] - out2))
        self.assertTrue(delta < 1e-3, f"fail at final output, delta {delta}")

        for i in range(config.num_hidden_layers):
            t1 = out1.hidden_states[i][0]
            t2 = layer_output[i]
            delta = torch.abs(torch.max(t2 - t1))
            self.assertTrue(delta < 1e-3, f"fail at layer {i}, delta {delta}")
