
from dataclasses import dataclass


@dataclass
class QwenConfig:
    hidden_size: int = 2048
    vocab_size: int = 151936
    layer_norm_epsilon: float = 1e-06
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 11008
    seq_length: int = 8192
    device: str = 'cuda'
    torch_dtype: str = 'float32'
    rotary_emb_base: int = 10000
