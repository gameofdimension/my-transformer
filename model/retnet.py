import torch
from torch import nn

from model.common import RMSNorm, Rotary
from model.retnet_config import RetnetConfig
from transformers.activations import get_activation


class Mlp(nn.Module):
    def __init__(self, config: RetnetConfig):
        super().__init__()
        self.w1 = nn.Linear(config.embed_dim, config.ffn_embed_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_embed_dim, config.embed_dim, bias=False)
        self.act_fn = get_activation(config.activation_fn)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.w1(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        hidden_state = self.w2(hidden_state)
        return hidden_state


class Retention(nn.Module):
    def __init__(self, config: RetnetConfig):
        super().__init__()
        self.config = config
        self.head_key_dim = config.embed_dim // config.retention_heads
        self.rotary = Rotary(self.head_key_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: float):
        bs, sq, _ = q.size()
        res = torch.zeros_like(v)
        for b in range(bs):
            for n in range(sq):
                qr = self.rotary.apply(n, q[b, n, :])
                for m in range(n + 1):
                    kr = self.rotary.apply(m, k[b, m, :])
                    res[b, n] += qr.dot(kr) * gamma ** (n - m) * v[b, m, :]
        return res


class MultiScaleRetention(nn.Module):
    def __init__(self, config: RetnetConfig):
        super().__init__()
        self.config = config
        self.gamma = [1 - 2 ** (-5 - i) for i in range(config.retention_heads)]
        assert config.embed_dim % config.retention_heads == 0
        self.head_key_dim = config.embed_dim // config.retention_heads
        self.head_value_dim = config.value_embed_dim // config.retention_heads
        self.group_norm = RMSNorm(self.head_value_dim, config.layernorm_eps)

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.value_embed_dim)
        self.g_proj = nn.Linear(config.embed_dim, config.value_embed_dim)
        self.o_proj = nn.Linear(config.value_embed_dim, config.embed_dim)

        self.retention = Retention(config)

    def forward(self, hidden_state: torch.Tensor):
        """

        :param hidden_state: [batch size, seq length, hidden size]
        :return:
        """
        bs, sq, hs = hidden_state.size()
        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)

        v = self.v_proj(hidden_state)
        g = get_activation('sigmoid')(self.g_proj(hidden_state))

        lst = []
        for h in range(self.config.retention_heads):
            base = h * self.head_key_dim
            base_v = h * self.head_value_dim
            qh = q[:, :, base:base + self.head_key_dim]
            kh = k[:, :, base:base + self.head_key_dim]
            vh = v[:, :, base_v:base_v + self.head_value_dim]
            msr = self.retention(qh, kh, vh, self.gamma[h])
            lst.append(msr.unsqueeze(dim=2))

        y = torch.concat(lst, dim=2)
        y = self.group_norm(y).reshape(bs, sq, -1)
        y = g * y
        x = self.o_proj(y)
        return x


class Block(nn.Module):
    def __init__(self, config: RetnetConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim, config.layernorm_eps)
        self.norm2 = RMSNorm(config.embed_dim, config.layernorm_eps)
        self.ffn = Mlp(config)
        self.retention = MultiScaleRetention(config)

    def forward(self, hidden_state: torch.Tensor):
        x = hidden_state
        x = self.norm1(x)
        x = self.retention(x)
        hidden_state += x

        x = hidden_state
        x = self.norm2(x)
        x = self.ffn(x)
        hidden_state += x

        return hidden_state


class Model(nn.Module):
    def __init__(self, config: RetnetConfig):
        super().__init__()
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.decoder_layers)])
        self.post_norm = RMSNorm(config.embed_dim, config.layernorm_eps)

    def forward(self, input_ids: torch.Tensor):
        """

        :param input_ids: [batch size, seq length]
        :return:
        """

        hidden_states = self.word_embedding_table(input_ids)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.post_norm(hidden_states), layers_output


def main():
    config = RetnetConfig(retention_heads=4)
    model = Model(config)

    a, b = model(torch.LongTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))
    print(a.size())


if __name__ == '__main__':
    main()
