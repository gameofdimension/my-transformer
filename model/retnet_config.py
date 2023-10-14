from dataclasses import dataclass


@dataclass
class RetnetConfig:
    embed_dim: int = 768
    value_embed_dim: int = 1280
    retention_heads: int = 3
    ffn_embed_dim: int = 1280
    decoder_layers: int = 12
    activation_fn: str = 'gelu'
    dropout: float = 0.0
    activation_dropout: float = 0.0
    layernorm_eps: float = 1e-5
    vocab_size: int = 64000
