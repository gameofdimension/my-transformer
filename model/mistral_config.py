
from dataclasses import dataclass


@dataclass
class MistralConfig:
    hidden_size: int = 4096
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-05
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    hidden_act: str = "silu"
    sliding_window: int = 4096
    max_position_embeddings: int = 32768
    device: str = 'cuda'
    rope_theta: float = 10000.0
