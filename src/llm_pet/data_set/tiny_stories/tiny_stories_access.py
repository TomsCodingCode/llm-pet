import re

import requests

from llm_pet.data_set.vocab import ALPHABET_SET, BOS_ID, EOS_ID, encode


# Greedy: everything up to a run of sentence-final punctuation (with any trailing
# closing quote/bracket), or the trailing fragment if a story has no final punctuation.
_SENTENCE_RE = re.compile(r'[^.!?]*[.!?]+(?:["\')\]]+)?|\S[^.!?]*$')

ROWS_PER_SHARD = 500  # stories held in RAM at once
VALIDATION_ROWS = 100  # fixed validation shard (kept constant across swaps)
HF_FETCH_CHUNK = 100  # datasets-server hard limit per request


def split_sentences(text):
    text = text.strip()
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


def story_to_examples(text, max_context):
    """One story -> list of (context_ids, decoder_in, decoder_target).

    For every sentence i >= 1:
      context     = sentences 0 .. i-1 joined with spaces
      decoder_in  = <bos> + sentence_i
      target      =         sentence_i + <eos>   (decoder_in shifted by one)

    The TARGET sentence must fit in max_context (we can't truncate what we are
    trying to generate), so any example whose target is too long is DISCARDED.
    The CONTEXT may come from an arbitrarily long story; we keep the most recent
    max_context tokens and truncate the rest. A whole story is therefore only
    useless if every one of its sentences is over-long.
    """
    sentences = split_sentences(text)
    examples = []
    for i in range(1, len(sentences)):
        full = [BOS_ID] + encode(sentences[i]) + [EOS_ID]  # length = sentence_len + 2

        # discard: decoder_in (full[:-1]) must fit within max_context positions
        if len(full) - 1 > max_context:
            continue

        # truncate: keep the most recent max_context tokens of prior sentences
        ctx_ids = encode(" ".join(sentences[:i]))[-max_context:]
        if len(ctx_ids) == 0:
            continue

        examples.append((ctx_ids, full[:-1], full[1:]))
    return examples


def get_tinystories_rows(split, offset, length, max_context):
    """Fetch `length` stories that (a) use only our alphabet and (b) yield at
    least one valid example (i.e. have at least one sentence whose target fits
    in max_context). Returns (rows, next_offset).

    Stories are filtered here too, so the shard we keep is already clean and we
    don't waste RAM on stories we'd discard entirely.
    """
    rows = []
    fetched = 0
    while len(rows) < length:
        url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=roneneldan%2FTinyStories&config=default"
            f"&split={split}&offset={offset + fetched}&length={HF_FETCH_CHUNK}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        batch = resp.json()["rows"]
        if not batch:
            break  # exhausted this split
        fetched += len(batch)
        for row in batch:
            text = row["row"]["text"]
            if not set(text).issubset(ALPHABET_SET):
                continue
            # keep only stories that produce at least one in-budget example
            if story_to_examples(text, max_context):
                rows.append(text)
                if len(rows) >= length:
                    break
    return rows[:length], offset + fetched
