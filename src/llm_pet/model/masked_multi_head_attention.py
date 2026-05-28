import torch
from torch import nn


class MaskedMultiHeadAttention(nn.Module):
    """Module for masked and unmasked self and cross attention."""

    def __init__(self, d_model, num_heads, mask, p_drop):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        layer_dim = self.d_head * num_heads

        self.query = nn.Linear(d_model, layer_dim)
        self.key = nn.Linear(d_model, layer_dim)
        self.value = nn.Linear(d_model, layer_dim)

        if mask is not None:
            assert mask.ndim == 4, "Mask must be a 4D tensor"
            self.register_buffer("mask", mask)
        else:
            self.mask = None
        self.dropout = nn.Dropout(p_drop)
        self.proj = nn.Linear(layer_dim, d_model)

    def forward(self, qs, ks=None, vs=None, key_padding_mask=None):
        # assume single x if not provided individually
        assert (ks is None) == (vs is None), (
            "Provide either qs alone (self-attention) or all of qs, ks, vs (cross-attention)"
        )
        if ks is None:
            ks = qs
        if vs is None:
            vs = qs

        # the query, key and value sources are of shape batch, by time, by channels
        B, T1, C = qs.shape
        B2, T2, C2 = ks.shape
        B3, T3, C3 = vs.shape
        assert T2 == T3, "Keys and values must have the same context length"
        assert C == C2 == C3, "Keys, queries and values must have the same channels"
        assert B == B2 == B3, "Batch size must be the same"

        # get queries, keys and values
        q = self.query(qs)  # B T1 C
        k = self.key(ks)  # B T2 C
        v = self.value(vs)  # B T2 C

        # unbatch to enable per head masked scaled dot product
        # (B, T, C) -> (B, T, H, C') -> (B, H, T, C')
        q = q.view(B, T1, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T2, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T2, self.num_heads, self.d_head).transpose(1, 2)

        # calculate affinities and apply masking
        affinity = q @ k.transpose(-1, -2) * self.d_head**-0.5  # (B, H, T1, T2)
        if self.mask is not None:
            affinity = torch.masked_fill(
                affinity, self.mask[:, :, :T1, :T2] == 0, float("-inf")
            )
        if key_padding_mask is not None:
            # (B, T2) -> (B, 1, 1, T2); broadcast over heads and query positions
            kpm = key_padding_mask[:, None, None, :T2]
            affinity = torch.masked_fill(affinity, kpm == 0, float("-inf"))

        # calculate new value
        weights = affinity.softmax(-1)
        weights = self.dropout(weights)
        out = weights @ v  # (B, H, T1, C')

        # restore shape and apply projection
        out = out.transpose(1, 2)  # (B, T1, H, C')
        out = out.reshape(B, T1, -1)
        out = self.proj(out)
        return out
