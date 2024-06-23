
from dataclasses import dataclass


@dataclass
class QwenConfig:
    hidden_size: int = 4096
    vocab_size: int = 151936
    layer_norm_epsilon: float = 1e-06
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 22016
    seq_length: int = 8192
    device: str = 'cuda'
    torch_dtype: str = 'float32'
    rotary_emb_base: int = 10000
