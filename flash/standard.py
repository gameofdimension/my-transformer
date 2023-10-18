import math

import torch


def vectorized_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = False):
    # [b,s,h,d]
    assert len(query.size()) == 4
    assert len(key.size()) == 4
    assert len(value.size()) == 4
    b, s, h, d = query.size()

    query = query.transpose(1, 2).reshape(b * h, s, d)
    key = key.transpose(1, 2).reshape(b * h, s, d).transpose(1, 2)
    value = value.transpose(1, 2).reshape(b * h, s, d)

    base = torch.zeros((1, 1, 1))
    scores = torch.baddbmm(input=base, batch1=query, batch2=key, beta=0) / math.sqrt(d)
    assert scores.size() == (b * h, s, s)
    if is_causal:
        attention_mask = torch.ones(1, s, s).tril() * 0.5
        attention_mask = attention_mask.masked_fill(attention_mask < 0.5, -float('inf'))
        attention_mask = attention_mask.masked_fill(attention_mask > 0.5, 0)
        scores += attention_mask
    probs = torch.softmax(scores, dim=-1)
    output = torch.baddbmm(input=base, batch1=probs, batch2=value, beta=0)

    output = output.reshape(b, h, s, d).transpose(1, 2)
    return output


def vectorized_attention_v2(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = False):
    # [b,s,h,d]
    assert len(query.size()) == 4
    assert len(key.size()) == 4
    assert len(value.size()) == 4
    b, s, h, d = query.size()

    query = query.transpose(1, 2)
    assert query.size() == (b, h, s, d)
    key = key.transpose(1, 2).transpose(2, 3)
    assert key.size() == (b, h, d, s)
    value = value.transpose(1, 2)
    assert value.size() == (b, h, s, d)

    scores = (query @ key) / math.sqrt(d)
    assert scores.size() == (b, h, s, s)
    if is_causal:
        attention_mask = torch.ones(1, 1, s, s).tril() * 0.5
        attention_mask = attention_mask.masked_fill(attention_mask < 0.5, -float('inf'))
        attention_mask = attention_mask.masked_fill(attention_mask > 0.5, 0)
        scores += attention_mask
    probs = torch.softmax(scores, dim=-1)
    assert probs.size() == (b, h, s, s)
    output = probs @ value
    return output.transpose(1, 2)
