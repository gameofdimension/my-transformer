from dataclasses import dataclass


@dataclass
class RwkvConfig:
    bos_token_id: int = 0
    eos_token_id: int = 0
    hidden_size: int = 768
    attention_hidden_size: int = 768
    intermediate_size: int = 3072
    context_length: int = 1024
    layer_norm_epsilon: float = 1e-05
    num_hidden_layers: int = 12
    rescale_every: int = 6
    vocab_size: int = 50277
