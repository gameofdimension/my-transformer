import torch
from torch import nn


def main():
    embedding = nn.Embedding(10, 3)
    print(embedding.weight[1])
    v1 = embedding(torch.LongTensor([1, 2, 3]))
    print(v1)


if __name__ == '__main__':
    main()
