from dataclasses import dataclass


@dataclass
class ChatGLMConfig:
    hidden_size: int = 4096
    vocab_size: int = 130528
    layernorm_epsilon: float = 1e-05
    num_layers: int = 28
    num_attention_heads: int = 32
    inner_hidden_size: int = 16384
    bos_token_id: int = 130004
    eos_token_id: int = 130005
    mask_token_id: int = 130000
    gmask_token_id: int = 130001
