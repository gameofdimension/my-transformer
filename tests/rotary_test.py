import unittest

import torch

import torch.nn.functional as F

from model.common import Rotary


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

    @staticmethod
    def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
        # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
        cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
                   F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
        q, k = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin), (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q, k


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (LlamaRotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (LlamaRotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed


class RotaryTest(unittest.TestCase):

    def test_gold(self):
        torch.manual_seed(0)
        # print()

        dim = 32
        gr = RotaryEmbedding(dim)
        lm = LlamaRotaryEmbedding(dim)

        num = 5
        position_ids = torch.LongTensor([[i for i in range(num)]])

        x1 = torch.normal(0, 1, (1, num, dim))
        y1 = torch.normal(0, 1, (1, num, dim))

        x2 = x1.clone().unsqueeze(2).transpose(0, 1)
        y2 = y1.clone().unsqueeze(2).transpose(0, 1)

        c1, s1 = lm(x=x2, seq_len=30)
        c2, s2 = gr(x=x1, seq_len=30)

        q1, k1 = LlamaRotaryEmbedding.apply_rotary_pos_emb(x1, y1, c1, s1, position_ids)
        q2, k2 = RotaryEmbedding.apply_rotary_pos_emb_index(x2, y2, c2, s2, position_ids.transpose(0, 1))

        self.assertTrue(torch.max(torch.abs(q1.squeeze() - q2.squeeze())) < 1e-3)
        self.assertTrue(torch.max(torch.abs(k1.squeeze() - k2.squeeze())) < 1e-3)

        my = Rotary(d=dim)
        x3 = x1.clone().squeeze()
        y3 = y1.clone().squeeze()
        for i in range(x3.size(0)):
            out = my.apply(i, x3[i])
            tou = my.apply(i, y3[i])
            self.assertTrue(torch.max(torch.abs(q1.squeeze()[i] - out.squeeze())) < 1e-3)
            self.assertTrue(torch.max(torch.abs(k1.squeeze()[i] - tou.squeeze())) < 1e-3)
