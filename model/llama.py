import torch
from torch import nn

from model.llama_config import LlamaConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.param = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        val = hidden_states * hidden_states
        norm = torch.sqrt(torch.mean(val, dim=-1, keepdim=True) + self.eps)
        return hidden_states / norm


class Block(nn.Module):
    pass


class Model(nn.Module):
    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # self.position_embedding_table = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        pass



