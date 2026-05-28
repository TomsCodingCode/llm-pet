import torch
import torch.utils.data

from llm_pet.data_set.tiny_stories.tiny_stories_access import story_to_examples
from llm_pet.data_set.vocab import PAD_ID


class TinyStoriesDataset(torch.utils.data.Dataset):
    """Flattens a list of stories into next-sentence examples."""

    def __init__(self, stories, max_context):
        self.examples = []
        for story in stories:
            self.examples.extend(story_to_examples(story, max_context))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ctx, dec_in, target = self.examples[idx]
        return (
            torch.tensor(ctx, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def _pad_stack(seqs, pad_value):
    longest = max(len(s) for s in seqs)
    out = torch.full((len(seqs), longest), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate(batch):
    # right-pad each field; build boolean padding masks (1 = real, 0 = pad)
    ctxs, dec_ins, targets = zip(*batch)
    context = _pad_stack(ctxs, PAD_ID)
    dec_in = _pad_stack(dec_ins, PAD_ID)
    target = _pad_stack(targets, PAD_ID)  # PAD positions ignored by the loss
    context_pad_mask = (context != PAD_ID).long()
    dec_in_pad_mask = (dec_in != PAD_ID).long()
    return context, dec_in, target, context_pad_mask, dec_in_pad_mask
