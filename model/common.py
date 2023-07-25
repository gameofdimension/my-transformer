from typing import Callable, List

import torch
from torch.nn.functional import softmax


def weighted_sum(scores: List[torch.Tensor], values: List[torch.Tensor]) -> torch.Tensor:
    v = torch.stack(values)
    scores = torch.Tensor(scores)
    probs = softmax(scores, dim=-1)
    return probs @ v


def attention_func(
        seq_length: int, num_attention_heads: int, hidden_size: int,
        get_q: Callable, get_k: Callable, get_v: Callable):
    assert hidden_size % num_attention_heads == 0
    head_dim = hidden_size // num_attention_heads
    scale = head_dim ** 0.5
    output = []
    for tk in range(seq_length):
        final_value = []
        for h in range(num_attention_heads):
            q = get_q(tk, h)
            s = []
            v = []
            for p in range(1 + tk):
                s.append(torch.dot(q, get_k(p, h)) / scale)
                v.append(get_v(p, h))
            final_value.append(weighted_sum(s, v))
        final_value = torch.stack(final_value).view(1, -1).squeeze()
        assert final_value.shape[0] == hidden_size
        output.append(final_value)
    output = torch.stack(output)
    return output
