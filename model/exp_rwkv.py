import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile")
    tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

    prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, " \
             "previously unexplored valley, in Tibet. Even more surprising to the researchers was " \
             "the fact that the dragons spoke perfect Chinese."

    inputs = tokenizer(prompt, return_tensors="pt")

    # print(inputs['input_ids'].size())
    xx = inputs['input_ids']
    xx = torch.concat([xx, xx, xx], dim=0)
    # output = model.generate(inputs["input_ids"], max_new_tokens=20)
    # output = model.generate(xx, max_new_tokens=20)
    #
    # print(type(output))
    #
    # print(tokenizer.decode(output[0].tolist()))
    # print(tokenizer.decode(output[1].tolist()))
    # print(tokenizer.decode(output[2].tolist()))

    input_ids = torch.LongTensor([[2, 5], [3, 4]])
    zz = model(input_ids, output_hidden_states=True)
    #
    # print(type(zz))
    print(len(zz.hidden_states))
    for hs in zz.hidden_states:
        print(hs.size())
        print(hs[:, :, :5])

    # for param in model.named_parameters():
    #     print(param[0], param[1].size())
