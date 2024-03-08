from dataclasses import dataclass


@dataclass
class GemmaConfig:
    hidden_size: int = 3072
    head_dim: int = 256
    vocab_size: int = 256000
    rms_norm_eps: float = 1e-06
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 24576
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8192
    bos_token_id: int = 2
    eos_token_id: int = 1
    rope_theta: float = 10000.0
    attention_bias: bool = False
    device: str = "cpu"
