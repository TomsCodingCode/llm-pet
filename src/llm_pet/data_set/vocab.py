BASE_ALPHABET = sorted(
    set(
        " \n!\"#$%&'()*/+,-.0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
    )
)
PAD_TOK, BOS_TOK, EOS_TOK = "<pad>", "<bos>", "<eos>"
SPECIALS = [PAD_TOK, BOS_TOK, EOS_TOK]

VOCAB = SPECIALS + BASE_ALPHABET
VOCAB_SIZE = len(VOCAB)

str_to_int = {ch: i for i, ch in enumerate(VOCAB)}
int_to_str = {i: ch for i, ch in enumerate(VOCAB)}

PAD_ID = str_to_int[PAD_TOK]
BOS_ID = str_to_int[BOS_TOK]
EOS_ID = str_to_int[EOS_TOK]

ALPHABET_SET = set(BASE_ALPHABET)  # used to reject out-of-vocab stories


def encode(s):
    # string -> list[int] over the base alphabet (no specials)
    return [str_to_int[c] for c in s]


def decode(ids):
    # list[int] -> string, dropping special tokens for readability
    out = []
    for i in ids:
        ch = int_to_str[int(i)]
        if ch in SPECIALS:
            continue
        out.append(ch)
    return "".join(out)
