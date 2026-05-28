import pandas as pd

from llm_pet.data_set.vocab import ALPHABET_SET, BOS_ID, EOS_ID, encode

GSM8K_PARQUET_PATHS = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

ROWS_PER_SHARD = 5000
VALIDATION_ROWS = 100
HF_FETCH_CHUNK = 100  # retained for compatibility; unused for local reads

# GSM8k (config "main") ships train (7473) and test (1319); no "validation"
# split, so we carve the held-out shard from "test". These are refreshed to the
# actual file sizes the first time each split is read (see _load_gsm8k_split).
GSM8K_SPLIT_SIZES = {"train": 7473, "test": 1319}

# Each Parquet file is read from disk once and cached as a list of
# (question, answer) tuples; subsequent shard requests just slice the cache.
_GSM8K_CACHE = {}


def question_to_example(question, answer, max_context):
    """One (question, answer) pair -> (context_ids, decoder_in, decoder_target)
    or None if it can't fit the budget.

      context     = question
      decoder_in  = <bos> + answer
      target      =         answer + <eos>   (decoder_in shifted by one)

    As in the TinyStories pipeline, the TARGET (the answer we must generate) has
    to fit within max_context positions, so over-long answers are DISCARDED.
    The CONTEXT (question) may be truncated to the most recent max_context
    tokens -- we keep the tail so the actual question being asked is preserved.
    """
    a = encode(answer)
    full = [BOS_ID] + a + [EOS_ID]  # length = answer_len + 2

    # discard: decoder_in (full[:-1]) must fit within max_context positions
    if len(full) - 1 > max_context:
        return None

    # truncate: keep the most recent max_context tokens of the question
    ctx_ids = encode(question)[-max_context:]
    if len(ctx_ids) == 0:
        return None

    return ctx_ids, full[:-1], full[1:]


def _load_gsm8k_split(split):
    """Read and cache one split's Parquet file as a list of (question, answer)."""
    if split not in _GSM8K_CACHE:
        path = GSM8K_PARQUET_PATHS[split]
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            # pandas needs pyarrow (or fastparquet) to read Parquet.
            raise ImportError(
                "Reading Parquet requires an engine -- run:  pip install pyarrow"
            ) from e
        rows = list(zip(df["question"].tolist(), df["answer"].tolist()))
        _GSM8K_CACHE[split] = rows
        GSM8K_SPLIT_SIZES[split] = len(rows)  # keep size in sync with the file
        print(f"[data] loaded {len(rows)} {split} rows from {path}")
    return _GSM8K_CACHE[split]


def get_gsm8k_rows(split, offset, length, max_context):
    """Return `length` filtered (question, answer) pairs from the local Parquet
    file for `split`, starting at raw-row `offset`. Returns (rows, next_offset).

    Mirrors the old streaming function exactly:
      * pairs are filtered here (alphabet-only AND in-budget) so the shard we
        keep is already clean;
      * `next_offset` advances by the number of RAW rows CONSUMED (not the number
        kept), and wraps at the end of the file, so repeated calls stream through
        the corpus the same way the HF reader did.

    Unlike the HF version this makes at most ONE full pass over the file, so a
    request for more clean rows than exist returns a short shard instead of
    looping forever -- the data module handles short shards fine.
    """
    corpus = _load_gsm8k_split(split)
    n = len(corpus)
    if n == 0:
        return [], offset

    rows = []
    consumed = 0
    i = offset % n
    while len(rows) < length and consumed < n:  # consumed < n => at most one pass
        question, answer = corpus[i]
        i = (i + 1) % n  # wrap to the top of the file
        consumed += 1
        if not set(question + answer).issubset(ALPHABET_SET):
            continue
        # keep only pairs that produce a valid, in-budget example
        if question_to_example(question, answer, max_context) is not None:
            rows.append((question, answer))
    return rows[:length], offset + consumed
