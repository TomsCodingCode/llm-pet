import torch
from torch import nn

from llm_pet.model.masked_multi_head_attention import MaskedMultiHeadAttention
from llm_pet.model.feed_forward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, max_context, p_drop):
        super().__init__()
        # layer norms
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.ln_3 = nn.LayerNorm(dim)

        # masked self attention
        mask = torch.tril(torch.ones(1, 1, max_context, max_context))
        self.sa_head = MaskedMultiHeadAttention(dim, num_heads, mask, p_drop)
        # unmasked cross attention
        self.ca_head = MaskedMultiHeadAttention(dim, num_heads, None, p_drop)
        # simple feed forward network
        self.ffw_net = FeedForward(dim, p_drop)

    def forward(
        self, x, context, x_key_padding_mask=None, context_key_padding_mask=None
    ):
        # self attention
        x = x + self.sa_head(self.ln_1(x), key_padding_mask=x_key_padding_mask)
        # cross attention
        x = x + self.ca_head(self.ln_2(x), context, context, context_key_padding_mask)
        x = x + self.ffw_net(self.ln_3(x))
        return x
