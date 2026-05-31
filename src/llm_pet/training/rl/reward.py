import re

import torch

from ...data_set.vocab import BOS_ID

# IMPORTANT — format markers.
# This model was fine-tuned on GSM8k, whose answers mark each calculator step
# with DOUBLE ANGLE BRACKETS "<<expr=result>>" and the final answer with FOUR
# hashes "#### <number>". That is what the model actually generates, so the
# parser below targets that format.
BLOCK_RE = re.compile(r"<<(.*?)>>")  # <<48/2=24>>  -> "48/2=24"
FINAL_RE = re.compile(r"####\s*(-?[\d,]+)")  # #### 72 -> "72"

# Reward weights (see the scheme in the section docstring below). Tweak here.
W_PRESENCE = 0.1  # reward per present block, up to the scored-block count
W_EXTRA = -0.1  # penalty per block BEYOND the scored count (anti-padding)
W_MATCH = 0.05  # reward per matching LEADING symbol vs the gold block
FINAL_BASE = 1.0  # reward for a correct final answer, regardless of reasoning
FINAL_PER_CORRECT = 0.5  # added per fully-correct scored block when final is right
KL_COEF = 0.1  # weight of the KL-to-reference penalty (see loss below)

# Small FORMAT reward: nudge the model to emit exactly one well-formed final
# "#### <number>" line. Kept tiny so it never competes with the accuracy signal;
# you can anneal it toward 0 once outputs are reliably parseable.
W_FORMAT_GOOD = 0.05  # +reward when there is EXACTLY one '#### n' line
W_FORMAT_BAD = -0.05  # penalty when there are zero, or more than one

# How many generated blocks earn presence/match reward (and beyond which count
# extra blocks are penalised as padding). The blocks-per-answer varies, so this
# is resolved PER QUESTION, not fixed:
#   "gold"  -> use the gold answer's own block count (default; correct steps are
#              rewarded however many there are, only true padding is punished)
#   "none"  -> score every generated block, never penalise (set W_EXTRA's effect off)
#   <int>   -> a fixed cap (e.g. 2 reproduces the original behaviour)
SCORED_BLOCKS_MODE = "gold"


def scored_block_count(gold_blocks):
    """Resolve how many blocks to score for THIS question (see SCORED_BLOCKS_MODE)."""
    if SCORED_BLOCKS_MODE == "gold":
        # at least 1 so a gold answer with no <<>> blocks still scores its first
        # generated block for presence rather than treating it as padding.
        return max(1, len(gold_blocks))
    if SCORED_BLOCKS_MODE == "none":
        return None  # None => "no cap", handled below
    return int(SCORED_BLOCKS_MODE)  # fixed integer cap


def extract_blocks(text):
    """Ordered list of reasoning-block CONTENTS, e.g. ['48/2=24', '48+24=72']."""
    return BLOCK_RE.findall(text)


def extract_final(text):
    """The last '#### <number>' value as a normalised string, or None."""
    matches = FINAL_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def count_final_lines(text):
    """How many '#### <number>' lines the text contains (for the format reward).
    Counts well-formed final-answer markers, so 0 or >1 both signal bad format."""
    return len(FINAL_RE.findall(text))


def _leading_match(gen_block, gold_block):
    """Count of matching leading characters of the two blocks, delimiters
    INCLUDED, matching the spec examples: '<<24>>' vs '<<24/2=12>>' shares the
    prefix '<<24' = 4; '<<24/>>' shares '<<24/' = 5. (Per the user's intent that
    '>>24<<' scores 2 and '>>24/<<' scores 3 over the digits/operators.)"""
    g = "<<" + gen_block + ">>"
    gd = "<<" + gold_block + ">>"
    n = 0
    for ca, cb in zip(g, gd):
        if ca == cb:
            n += 1
        else:
            break
    return n


def compute_reward(gen_text, gold_text):
    """Scalar reward for one generated answer against its gold answer.

    Four additive components:

      1. PRESENCE  -- each generated block up to the per-question scored count
         (see `scored_block_count`) earns `W_PRESENCE`; every block BEYOND that
         count earns `W_EXTRA` (negative), so padding the answer with extra
         <<...>> blocks is punished. The scored count tracks the gold answer's
         own block count by default, so answers with three or four legitimate
         steps reward all of them and only penalise genuine padding.

      2. MATCH     -- for each scored generated block, a larger reward of
         `W_MATCH` per matching LEADING symbol against the corresponding gold
         block (longest common prefix, delimiters included).

      3. FINAL     -- given purely on the final '#### n' matching the gold final,
         regardless of reasoning: `FINAL_BASE` (1.0). It is then increased by
         `FINAL_PER_CORRECT` (0.5) for each scored block that is FULLY correct
         (e.g. 1.0 / 1.5 / 2.0 for none / one / two correct, and higher when the
         gold answer has more scored steps).

      4. FORMAT    -- a small `W_FORMAT_GOOD` when the answer has EXACTLY one
         well-formed '#### n' line, else `W_FORMAT_BAD`. Tiny by design: it only
         shapes the output toward parseability and never rivals the accuracy term.
    """
    gen_blocks = extract_blocks(gen_text)
    gold_blocks = extract_blocks(gold_text)
    gen_final = extract_final(gen_text)
    gold_final = extract_final(gold_text)

    # per-question: how many blocks are "scored"; None => no cap (score all).
    n_scored = scored_block_count(gold_blocks)
    cap = len(gen_blocks) if n_scored is None else n_scored

    reward = 0.0

    # 1) presence / padding
    n_present = min(len(gen_blocks), cap)
    n_extra = 0 if n_scored is None else max(0, len(gen_blocks) - cap)
    reward += W_PRESENCE * n_present + W_EXTRA * n_extra

    # 2) leading-symbol match on the scored blocks, vs the SAME-INDEX gold block
    for i in range(min(cap, len(gen_blocks))):
        gold_b = gold_blocks[i] if i < len(gold_blocks) else ""
        reward += W_MATCH * _leading_match(gen_blocks[i], gold_b)

    # 3) final answer (reasoning-independent base, scaled by fully-correct blocks)
    if gen_final is not None and gold_final is not None and gen_final == gold_final:
        n_correct = 0
        for i in range(min(cap, len(gen_blocks))):
            if i < len(gold_blocks) and gen_blocks[i].strip() == gold_blocks[i].strip():
                n_correct += 1
        reward += FINAL_BASE + FINAL_PER_CORRECT * n_correct

    # 4) format: exactly one '#### n' line is good; zero or many is bad
    reward += W_FORMAT_GOOD if count_final_lines(gen_text) == 1 else W_FORMAT_BAD

    return reward


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
