from torch import nn

from llm_pet.model.masked_multi_head_attention import MaskedMultiHeadAttention
from llm_pet.model.feed_forward import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, max_context, p_drop):
        super().__init__()
        # layer norms
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

        # unmasked self attention
        self.sa_head = MaskedMultiHeadAttention(dim, num_heads, None, p_drop)
        # simple feed forward network
        self.ffw_net = FeedForward(dim, p_drop)

    def forward(self, x, key_padding_mask=None):
        x = x + self.sa_head(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffw_net(self.ln_2(x))
        return x
