import torch
from torch import nn

from llm_pet.model.encoder_block import EncoderBlock
from llm_pet.model.decoder_block import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_context, vocab_size, p_drop):
        super().__init__()
        self.max_context = max_context
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_context, dim)

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(dim, num_heads, max_context, p_drop)
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(dim, num_heads, max_context, p_drop)
                for _ in range(num_layers)
            ]
        )

        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x, context, x_pad_mask=None, context_pad_mask=None):
        assert x.shape[1] <= self.max_context, "Input sequence too long"
        assert context.shape[1] <= self.max_context, "Context sequence too long"

        # generate context embeddings first
        tok = self.tok_emb(context.long())
        pos = self.pos_emb(torch.arange(context.shape[1], device=context.device))
        context = tok + pos
        for block in self.encoder:
            context = block(context, context_pad_mask)

        # generate target embeddings
        tok = self.tok_emb(x.long())
        pos = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = tok + pos

        # perform decoder pass
        for block in self.decoder:
            x = block(x, context, x_pad_mask, context_pad_mask)

        # project embedding into vocab dimension
        logits = self.proj(x)
        return logits

    def generate(self, context, start_tokens, max_tokens):
        idx = start_tokens
        for _ in range(max_tokens):
            logits = self(idx[:, -self.max_context :], context[:, -self.max_context :])
            logits = logits[:, -1, :]
            probs = logits.softmax(dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
