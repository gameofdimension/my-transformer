from dataclasses import dataclass


@dataclass
class Baichuan13bConfig:
    hidden_size: int = 5120
    vocab_size: int = 64000
    rms_norm_eps: float = 1e-06
    num_hidden_layers: int = 40
    num_attention_heads: int = 40
    intermediate_size: int = 13696
    hidden_act: str = "silu"
