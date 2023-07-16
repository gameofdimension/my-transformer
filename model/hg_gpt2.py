import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoConfig


def main():
    config = AutoConfig.from_pretrained('gpt2')
    # print(type(config), config)
    # return

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    obj = model.state_dict()
    for name in obj:
        print(name)

    obj = model.named_parameters()
    for params in obj:
        print(params[0])
    return

    model.eval()

    prompts = ["can i", "it's hot outside"]
    tks = tokenizer(prompts)
    print("raw tokens", tks)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collated = data_collator(tks['input_ids'])
    print("collated", collated)

    with torch.no_grad():
        output_ids = model(torch.LongTensor(collated['input_ids']))
    output_ids = output_ids.logits.argmax(dim=-1)
    print(output_ids)

    print(tokenizer.decode(output_ids[0]))
    print(tokenizer.decode(output_ids[1]))


def exp_embedding():
    embedding = nn.Embedding(10, 3)
    print(embedding.weight[1])
    v1 = embedding(torch.LongTensor([1, 2, 3]))
    print(v1)


if __name__ == '__main__':
    main()
    # test_embedding()
