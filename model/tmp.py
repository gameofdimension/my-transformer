import torch

ScalingWeight = Float[torch.Tensor, f"positiveEntries={N_EMBD}"]  # positive number, one per channel


class WKVMemory(torch.nn.Module):
    """A memory module whose contents exponentially decay over time, at a different rate per channel."""

    def __init__(self):
        super().__init__()

        # learned memory parameters -- one value for each dimension in the embeddings
        self.log_gain: ChannelParameter = torch.nn.Parameter(torch.zeros(N_EMBD))
        self.log_decay: ChannelParameter = torch.nn.Parameter(torch.zeros(N_EMBD))

        # state buffers to track information across a sequence
        contents, normalizer = torch.zeros(N_EMBD), torch.zeros(N_EMBD)
        self.register_buffer("contents", contents, persistent=False)
        self.register_buffer("normalizer", normalizer, persistent=False)

    @beartype
    def update(self, importances: ScalingWeight, values: Embedding) -> Tuple[Update, Update]:
        """Updates the memory by incrementing time and mixing in the weighted input values."""
        # decay the information currently in memory by one step
        self.step()

        # compute new information to add to the memory
        contents_update: Update = importances * values  # scale each value by the matching importance weight
        normalizer_update: Update = importances  # keep track of the weights so we can normalize across steps

        # and then add the new information to the memory
        self.contents += contents_update
        self.normalizer += normalizer_update  # -- including updating the normalizer!

        # and return it
        return contents_update, normalizer_update

    def step(self):
        """Pushes the information currently in the memory towards zero."""
        decay_rate: ScalingWeight = exp(self.log_decay)  # exp ensures that decay rate is positive
        self.contents *= exp(-decay_rate)  # decay_rate > 0, so exp(-decay_rate) < 1
        self.normalizer *= exp(-decay_rate)  # so each .step shrinks the contents and normalizer towards 0

    def apply_gain(self, latest_contents, latest_normalizer):
        """Applies the channelwise gain to the latest contents and normalizer."""
        gain = exp(self.log_gain) - 1  # -1 < gain < inf

        boosted_contents = gain * latest_contents
        boosted_normalizer = gain * latest_normalizer

        return boosted_contents, boosted_normalizer

    @beartype
    def forward(self, values: Embedding, importances: ScalingWeight) -> Update:
        """Applies the RWKV "time-mixing block" forward pass, in the "RNN Cell" style.

        For details, see https://arxiv.org/abs/2305.13048, Appendix B, Eqn. 19-22 and Fig. 7."""
        # first, we update the memory and return what we just added
        latest_contents, latest_normalizer = self.update(importances, values)

        # then, we adjust the representation of the latest information
        latest_contents, latest_normalizer = self.apply_gain(latest_contents, latest_normalizer)

        # before adding it in and dividing, to get the final thing we report as output
        out: Update = (self.contents + latest_contents) / \
                      (self.normalizer + latest_normalizer)

        return out


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # linear operations
        self.key = torch.nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.value = torch.nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.receptance = torch.nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.output = torch.nn.Linear(N_EMBD, N_EMBD, bias=False)

        # mixers
        self.key_mixer, self.value_mixer = Mixer(), Mixer()
        self.receptance_mixer = Mixer()

        # memory
        self.memory: torch.nn.Module = WKVMemory()

    @beartype
    def forward(self, x: Embedding) -> Update:
        # as with the MLP, do mixers before anything else
        mixed_keys = self.key_mixer(x)
        keys: Embedding = self.key(mixed_keys)

        mixed_values = self.value_mixer(x)
        values: Embedding = self.value(mixed_values)

        # wkv: apply "w"eighted decay to merge
        #      current info ("k"eys and "v"alues) with past
        wkv: Embedding = self.memory(values, exp(keys))

        # decide how "r"eceptive each channel is to inputs
        mixed_receptances = self.receptance_mixer(x)
        receptances: Embedding = self.receptance(mixed_receptances)
        gating_values = sigmoid(receptances)

        # rwkv: use the "r"eceptances to gate the output of the "wkv" memory
        rwkv: Embedding = gating_values * wkv

        # and then do one final linear transform before returning it
        dx: Update = self.output(rwkv)

        return dx
