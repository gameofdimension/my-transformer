import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile")
    tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

    prompt = "\nonce upon a"

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs["input_ids"], max_new_tokens=40)
    print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

    input_ids = torch.LongTensor([[2, 5], [3, 4]])
    output = model(input_ids, output_hidden_states=True)
    for hs in output.hidden_states:
        print(hs.size())
        print(hs[:, :, :5])


if __name__ == '__main__':
    main()
