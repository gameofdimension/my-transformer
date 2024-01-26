from dataclasses import dataclass


@dataclass
class DeepseekConfig:
    vocab_size: int = 102400
    hidden_act: str = "silu"
    hidden_size: int = 2048
    intermediate_size: int = 10944
    max_position_embeddings: int = 4096
    moe_intermediate_size: int = 1408
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_attention_heads: int = 16
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = False
    num_hidden_layers: int = 28
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-06
    attention_bias: bool = False
    scoring_func: str = 'softmax'
    rope_theta: int = 10000
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    device: str = 'cuda'
    torch_dtype: str = 'bfloat16'
