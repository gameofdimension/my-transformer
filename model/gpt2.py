import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import get_activation

from model.common import attention_func
from model.gpt2_config import Gpt2Config


class Mlp(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_fn = get_activation(config.activation_function)

        self.up_proj = nn.Linear(in_features=config.n_embd, out_features=4 * config.n_embd)
        self.down_proj = nn.Linear(in_features=4 * config.n_embd, out_features=config.n_embd)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        return self.down_proj(self.action_fn(self.up_proj(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config.n_embd % config.n_head == 0
        dim = config.n_embd // config.n_head
        self.qkv_dim = 3 * dim
        self.qkv_proj = nn.Linear(config.n_embd, self.qkv_dim * config.n_head)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """

        all_head_qkv = self.qkv_proj(hidden_states)
        dim = self.qkv_dim // 3
        step = all_head_qkv.size(dim=-1) // 3

        def get_q(idx: int, head: int):
            base = step * 0
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        def get_k(idx: int, head: int):
            base = step * 1
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        def get_v(idx: int, head: int):
            base = step * 2
            return all_head_qkv[idx, base + head * dim:base + (head + 1) * dim]

        seq_length = hidden_states.shape[0]
        output = attention_func(
            seq_length=seq_length, num_attention_heads=self.config.n_head,
            hidden_size=self.config.n_embd, get_q=get_q, get_k=get_k, get_v=get_v)
        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ln1 = nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mha = MultiHeadAttention(config)

        self.ln2 = nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = Mlp(config)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        for f, g in zip([self.mha, self.mlp], [self.ln1, self.ln2]):
            gx = g(x)
            fx = f(gx)
            x = x + fx
        return x


def name_mapping(param: str):
    out = {
        "word_embedding_table.weight": "transformer.wte.weight",
        "position_embedding_table.weight": "transformer.wpe.weight",
        "ln.weight": "transformer.ln_f.weight",
        "ln.bias": "transformer.ln_f.bias",
    }
    if param in out:
        return out[param]

    li = param.split('.')[1]
    prefix = f"transformer.h.{li}."
    if "ln1.weight" in param:
        postfix = "ln_1.weight"
    elif "ln1.bias" in param:
        postfix = "ln_1.bias"
    elif "mha.qkv_proj.weight" in param:
        postfix = "attn.c_attn.weight"
    elif "mha.qkv_proj.bias" in param:
        postfix = "attn.c_attn.bias"
    elif "mha.out_proj.weight" in param:
        postfix = "attn.c_proj.weight"
    elif "mha.out_proj.bias" in param:
        postfix = "attn.c_proj.bias"
    elif "ln2.weight" in param:
        postfix = "ln_2.weight"
    elif "ln2.bias" in param:
        postfix = "ln_2.bias"
    elif "mlp.up_proj.weight" in param:
        postfix = "mlp.c_fc.weight"
    elif "mlp.up_proj.bias" in param:
        postfix = "mlp.c_fc.bias"
    elif "down_proj.weight" in param:
        postfix = "mlp.c_proj.weight"
    elif "down_proj.bias" in param:
        postfix = "mlp.c_proj.bias"
    else:
        assert False

    return prefix + postfix


class Model(nn.Module):
    def __init__(self, config: Gpt2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        """
        单序列玩具版 gpt2，纯手搓
        不支持 batch 输入
        :param input_ids: [seq_length]
        :return:
        """
        assert len(input_ids.shape) == 1
        position_ids = torch.LongTensor(range(input_ids.shape[0]))

        word_embeddings = self.word_embedding_table(input_ids)
        position_embeddings = self.position_embedding_table(position_ids)

        hidden_states = word_embeddings + position_embeddings
        layers_output = [hidden_states.detach()]

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())

        return self.ln(hidden_states), layers_output

    def load_weights_from_hf(self, model_id):
        """
        因为是复刻 huggingface gpt2，所以可以直接加载其模型权重
        :return:
        """
        # model_id = 'gpt2'
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            if "weight" in name and ('.mlp.' in name or '.mha.' in name):
                # gpt2 用了 Conv1D 而不是 nn.Linear
                param.data.copy_(ref_param.transpose(0, 1))
            else:
                param.data.copy_(ref_param)


def count_params():
    """
    设层数为 L，隐层大小为 H，注意力头数为 T，Q,K,V 的维数为 D。词表大小 C，最大支持序列长度 S
    1. 每层有两个 layer normalization 层，隐层的每个成员对应 2 个参数，所以这部分参数有 2*2H
    2. 每层的 MLP 首先将 H 升为 4H，然后将 4H 降为 H，加上各自的 bias 共有参数 H*4H+4H+4H*H+H
    3.1 多头注意力部分，因为需要提供 T 个注意力头的 Q,K,V 所以，输出维度为 3D*T
    3.2 每个头的输出是其他头的 V 的加权和，所以维度也为 D，同时所有这样的输出拼起来维度需为 H，说明 TD = H
    3.3 这部分的参数数为 H*(3D*T)+3D*T ，也就是 H*3H+3H
    3.4 多头注意层还有个全连接层参数量为 H*H+H
    4. 综上每层的参数为 12*H^2+13*H
    5.1 除各层参数之外还有词表 embedding 查询表，参数量为 C*H
    5.2 位置 embedding 查询表 S*H
    5.3 最后的输出还会被 layer normalize ，对应参数 2H
    6. 总参数量为 (12*H^2+13*H)*L+C*H+S*H+2*H。
    7. huggingface gpt2 模型的 L 为 12，H 为 768，D 为 64，C 为 50257，S 为 1024
    :return:
    """
    L = 12
    H = 768
    C = 50257
    S = 1024

    num_per_layer = 12 * H ** 2 + 13 * H
    num = num_per_layer * L + C * H + S * H + 2 * H
    print(num_per_layer, num)

    def query_model_params(m):
        total = 0
        for n, p in m.named_parameters():
            total += p.numel()
        return total

    config = Gpt2Config()
    my = Model(config)
    assert num == query_model_params(my)

    gold = AutoModelForCausalLM.from_pretrained("gpt2")
    assert num == query_model_params(gold)


if __name__ == '__main__':
    count_params()
