from dataclasses import dataclass


@dataclass
class ChatGLM2Config:
    hidden_size: int = 4096
    padded_vocab_size: int = 65024
    layernorm_epsilon: float = 1e-05
    num_layers: int = 28
    num_attention_heads: int = 32
    ffn_hidden_size: int = 13696
    kv_channels: int = 128
    multi_query_attention: bool = True
    multi_query_group_num: int = 2


"""
"add_bias_linear": false,
  "add_qkv_bias": true,
  "apply_query_key_layer_scaling": true,
  "apply_residual_connection_post_layernorm": false,
  "attention_dropout": 0.0,
  "attention_softmax_in_fp32": true,
  "bias_dropout_fusion": true,
  "ffn_hidden_size": 13696,
  "fp32_residual_connection": false,
  "hidden_dropout": 0.0,
  "hidden_size": 4096,
  "kv_channels": 128,
  "layernorm_epsilon": 1e-05,
  "multi_query_attention": true,
  "multi_query_group_num": 2,
  "num_attention_heads": 32,
  "num_layers": 28,
  "original_rope": true,
  "padded_vocab_size": 65024,
  "post_layer_norm": true,
  "rmsnorm": true,
  "seq_length": 32768,
  "use_cache": true,
  "torch_dtype": "float16",
  "transformers_version": "4.27.1",
  "tie_word_embeddings": false,
  "eos_token_id": 2,
  "pad_token_id": 0
}
"""
