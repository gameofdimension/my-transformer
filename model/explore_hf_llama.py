from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

import torch


def main():
    config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
    print(config)
    config.num_hidden_layers = 4
    model = LlamaForCausalLM(config)
    for params in model.named_parameters():
        print(params[0], params[1].size())


def slim():
    my_model = AutoModelForCausalLM.from_pretrained("felixdae/Llama-2-7b-hf")
    out = my_model(input_ids=torch.LongTensor([[42]]), output_hidden_states=True)
    print(len(out.hidden_states))
    print(out.hidden_states[0].size())
    print(out.hidden_states[1].size())
    print(out.hidden_states[2].size())


if __name__ == '__main__':
    slim()
