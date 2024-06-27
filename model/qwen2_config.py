
from dataclasses import dataclass


@dataclass
class Qwen2Config:
    hidden_size: int = 1536
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-06
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    max_position_embeddings: int = 2048
    device: str = 'cuda'
    torch_dtype: str = 'float32'
    rope_theta: float = 1000000.0
    use_sliding_window: bool = False
    hidden_act: str = 'silu'
