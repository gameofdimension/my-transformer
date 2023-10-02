import torch
from torch import nn
from transformers import AutoModelForCausalLM

from model.rwkv_config import RwkvConfig


class Mixer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.empty(1, 1, hidden_size))

    def forward(self, current: torch.Tensor, previous: torch.Tensor):
        # [batch size, seq length, hidden size]
        assert len(current.size()) == 3

        mixed = self.mix_weight * current + (1 - self.mix_weight) * previous
        return mixed


class FeedForward(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.key_mixer = Mixer(config.hidden_size)
        self.receptance_mixer = Mixer(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        current = hidden_states
        # shift along time dimension
        previous = self.time_shift(current)

        key = self.key(self.key_mixer(current, previous))
        key = torch.square(torch.relu(key))
        value = self.value(key)

        receptance = torch.sigmoid(self.receptance(self.receptance_mixer(current, previous)))
        return receptance * value


class Memory(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        self.time_decay = nn.Parameter(torch.empty(config.attention_hidden_size))
        self.time_first = nn.Parameter(torch.empty(config.attention_hidden_size))

    def forward(self, key: torch.Tensor, value: torch.Tensor):
        batch_size, seq_length, hidden_size = key.size()
        importance = torch.exp(key)
        time_decay = -torch.exp(self.time_decay)

        lst = []
        a = torch.zeros(batch_size, hidden_size)
        b = torch.zeros(batch_size, hidden_size)
        for i in range(seq_length):
            w = torch.exp(self.time_first) * importance[:, i]
            vt = value[:, i]
            wkv = (a + w * vt) / (b + w)
            a = torch.exp(time_decay) * a + importance[:, i] * vt
            b = torch.exp(time_decay) * b + importance[:, i]

            lst.append(wkv.unsqueeze(1))
        return torch.concat(lst, dim=1)


class Attention(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        attention_hidden_size = config.attention_hidden_size
        hidden_size = config.hidden_size

        self.key_mixer = Mixer(config.hidden_size)
        self.value_mixer = Mixer(config.hidden_size)
        self.receptance_mixer = Mixer(config.hidden_size)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)

        self.memory = Memory(config)

    def forward(self, hidden_states: torch.Tensor):
        assert len(hidden_states.size()) == 3
        current = hidden_states
        # shift along time dimension
        previous = self.time_shift(current)

        key = self.key(self.key_mixer(current, previous))
        value = self.value(self.value_mixer(current, previous))
        receptance = torch.sigmoid(self.receptance(self.receptance_mixer(current, previous)))

        rwkv = self.memory(key, value)
        output = self.output(receptance * rwkv)
        return output


class Block(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = Attention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ffn = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor):
        attention = self.attention(self.ln1(hidden_states))
        hidden_states = hidden_states + attention

        feed_forward = self.ffn(self.ln2(hidden_states))
        hidden_states = hidden_states + feed_forward
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        # self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        """
        :param input_ids: shape [batch_size, seq_length]
        :return:
        """
        assert len(input_ids.size()) == 2
        hidden_states = self.word_embedding_table(input_ids)
        hidden_states = self.pre_ln(hidden_states)
        layers_output = [hidden_states.detach().clone()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach().clone())
        return self.post_ln(hidden_states), layers_output

    def load_weights_from_hf(self, model_id):
        """
        :return:
        """
        # "RWKV/rwkv-4-169m-pile"
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)


def name_mapping(param: str):
    out = {
        "word_embedding_table.weight": "rwkv.embeddings.weight",
        "pre_ln.weight": "rwkv.blocks.0.pre_ln.weight",
        "pre_ln.bias": "rwkv.blocks.0.pre_ln.bias",
        "post_ln.weight": "rwkv.ln_out.weight",
        "post_ln.bias": "rwkv.ln_out.bias",
    }
    if param in out:
        return out[param]

    arr = param.split('.')
    assert arr[0] == 'layers'
    layer_id = arr[1]
    sub = arr[2]

    prefix = f"rwkv.blocks.{layer_id}"
    if sub in ['ln1', 'ln2']:
        return prefix + "." + sub + "." + arr[-1]
    if sub == 'attention':
        if 'time_decay' in param or 'time_first' in param:
            return prefix + "." + sub + "." + arr[-1]
        if 'key_mixer' in param:
            return prefix + "." + sub + ".time_mix_key"
        if 'value_mixer' in param:
            return prefix + "." + sub + ".time_mix_value"
        if 'receptance_mixer' in param:
            return prefix + "." + sub + ".time_mix_receptance"
        return prefix + "." + sub + f".{arr[-2]}.{arr[-1]}"
    if sub == 'ffn':
        if 'key_mixer' in param:
            return prefix + ".feed_forward.time_mix_key"
        if 'receptance_mixer' in param:
            return prefix + ".feed_forward.time_mix_receptance"
        return prefix + f".feed_forward.{arr[-2]}.{arr[-1]}"


def main():
    model = Model(RwkvConfig())
    model_id = "RWKV/rwkv-4-169m-pile"
    model.load_weights_from_hf(model_id)

    input_ids = torch.LongTensor([[2, 5], [3, 4]])
    hs, output = model(input_ids)
    print(hs.size())
    print(hs[:, :, :5])


"""
tensor([[[ 0.3331, -0.1893, -0.9371, -0.2626,  0.2004],
         [ 0.3663, -0.1645, -0.9951, -0.3056,  0.0879]],

        [[-0.0356,  0.0923, -0.0142, -0.3919,  0.2644],
         [-0.0784,  0.1401, -0.5753, -0.4224,  0.2791]]],
       grad_fn=<SliceBackward0>)
"""


if __name__ == '__main__':
    main()
