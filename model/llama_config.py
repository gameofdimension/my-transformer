from dataclasses import dataclass


@dataclass
class LlamaConfig:
    hidden_size: int = 4096
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-05
    num_hidden_layers: int = 2
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    hidden_act: str = "silu"
