
from dataclasses import dataclass


@dataclass
class MistralConfig:
    hidden_size: int = 4096
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-05
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    device: str = 'cuda'
    torch_dtype: str = 'float32'
    rope_theta: float = 10000.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
