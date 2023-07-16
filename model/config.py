from dataclasses import dataclass


@dataclass
class Gpt2Config:
    """
    reference transformers.GPT2Config
    """
    activation_function: str = "gelu_new"

    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    vocab_size: int = 50257

    layer_norm_epsilon: float = 1e-05
