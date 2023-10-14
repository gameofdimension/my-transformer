import torch
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

if __name__ == '__main__':
    config = RetNetConfig(vocab_size=64000, decoder_retention_heads=4)
    token_embeddings = torch.nn.Embedding(config.vocab_size, config.decoder_embed_dim)
    retnet = RetNetDecoder(config, embed_tokens=token_embeddings)

    print(retnet)
    retnet(torch.LongTensor([[1, 2, 3]]))
