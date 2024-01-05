from dataclasses import dataclass


@dataclass
class Phi2Config:
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-05
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    vocab_size: int = 51200
    n_embd: int = 2560
    n_head: int = 32
    n_layer: int = 32
    n_positions: int = 2048
    rotary_dim: int = 32
    device: str = 'cuda'
