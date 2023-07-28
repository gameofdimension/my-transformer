import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-430m-pile")
    tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-430m-pile")

    prompt = "\nhello"

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs["input_ids"], max_new_tokens=40)
    print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))


if __name__ == '__main__':
    main()
