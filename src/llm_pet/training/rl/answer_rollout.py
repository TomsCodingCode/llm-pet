import torch

from llm_pet.data_set.vocab import BOS_ID, EOS_ID, decode


@torch.enable_grad()
def rollout(policy, context_ids, max_tokens, sample=True):
    """Generate one answer for a single question.

    Args:
        policy:       the SLMLightning module being trained (its .model is the
                      encoder-decoder Transformer).
        context_ids:  list[int] of the (already tail-truncated) question tokens.
        max_tokens:   decode budget (<= 128 because of the fixed position table).
        sample:       True -> multinomial sampling (needed for exploration);
                      False -> greedy argmax (used for eval).

    Returns dict with:
        text      : decoded answer string (specials stripped)
        token_ids : LongTensor (L,) of generated tokens (excluding the <bos>)
        logps     : FloatTensor (L,) log-prob of each generated token under policy
        mask      : FloatTensor (L,) 1.0 for tokens up to & including first <eos>
    """
    mc = policy.hparams.max_context
    device = policy.device
    context = torch.tensor([context_ids[-mc:]], device=device)  # (1, Tc)
    idx = torch.tensor([[BOS_ID]], device=device)  # (1, 1)

    logps, toks = [], []
    finished = False
    for _ in range(max_tokens):
        logits = policy.model(idx[:, -mc:], context[:, -mc:])  # (1, t, V)
        logits = logits[:, -1, :]  # (1, V)
        logprobs = torch.log_softmax(logits, dim=-1)  # (1, V)
        if sample:
            nxt = torch.multinomial(logprobs.exp(), num_samples=1)  # (1, 1)
        else:
            nxt = logprobs.argmax(dim=-1, keepdim=True)  # (1, 1)
        logps.append(logprobs.gather(-1, nxt).squeeze(1))  # (1,)
        toks.append(nxt.squeeze(1))  # (1,)
        idx = torch.cat([idx, nxt], dim=1)
        if int(nxt.item()) == EOS_ID:
            finished = True
            break

    token_ids = (
        torch.cat(toks) if toks else torch.zeros(0, dtype=torch.long, device=device)
    )
    logps = torch.cat(logps) if logps else torch.zeros(0, device=device)

    # mask: everything up to & including the first <eos> is "real"; here the loop
    # already stops at <eos>, so all sampled positions count.
    mask = torch.ones_like(logps)

    gen = token_ids.tolist()
    if EOS_ID in gen:
        gen = gen[: gen.index(EOS_ID)]
    text = decode(gen)
    return {
        "text": text,
        "token_ids": token_ids,
        "logps": logps,
        "mask": mask,
        "finished": finished,
    }


@torch.no_grad()
def reference_logps(ref_policy, context_ids, token_ids):
    """Log-probs of an ALREADY-SAMPLED token sequence under the frozen reference
    model -- used for the KL penalty. Teacher-forced in one pass over the tokens
    the policy actually produced."""
    mc = ref_policy.hparams.max_context
    device = ref_policy.device
    context = torch.tensor([context_ids[-mc:]], device=device)
    # decoder input is <bos> + all but the last produced token
    dec_in = torch.tensor([[BOS_ID] + token_ids.tolist()[:-1]], device=device)[:, :mc]
    logits = ref_policy.model(dec_in, context)  # (1, L, V)
    logprobs = torch.log_softmax(logits, dim=-1)[0]  # (L, V)
    L = min(len(token_ids), logprobs.size(0))
    idx = token_ids[:L].unsqueeze(1)
    return logprobs[:L].gather(-1, idx).squeeze(1)  # (L,)
