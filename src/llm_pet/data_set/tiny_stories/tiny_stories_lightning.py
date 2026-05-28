import pytorch_lightning as pl
import torch
import torch.utils.data

from llm_pet.data_set.tiny_stories.tiny_stories_dataset import (
    TinyStoriesDataset,
    collate,
)
from llm_pet.data_set.tiny_stories.tiny_stories_access import (
    get_tinystories_rows,
    ROWS_PER_SHARD,
    VALIDATION_ROWS,
)


class TinyStoriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        rows_per_shard=ROWS_PER_SHARD,
        validation_rows=VALIDATION_ROWS,
        batch_size=64,
        max_context=128,
        seed=42,
    ):
        super().__init__()
        self.rows_per_shard = rows_per_shard
        self.validation_rows = validation_rows
        self.batch_size = batch_size
        self.max_context = max_context

        # torch generator for picking the (fixed) validation offset
        self._gen = torch.Generator().manual_seed(seed)
        self._train_offset = 0  # advances as we stream through the corpus
        self.train_dataset = None
        self.val_dataset = None
        self._shard_index = 0

    def setup(self, stage=None):
        if self.val_dataset is None:
            # fixed random validation slice so val loss is comparable across swaps
            high = max(1, 21000 - self.validation_rows)
            val_offset = int(torch.randint(0, high, (1,), generator=self._gen).item())
            val_rows, _ = get_tinystories_rows(
                "validation", val_offset, self.validation_rows, self.max_context
            )
            self.val_dataset = TinyStoriesDataset(val_rows, self.max_context)
            print(
                f"[data] validation shard: {len(val_rows)} stories "
                f"-> {len(self.val_dataset)} examples"
            )
        if self.train_dataset is None:
            self.load_new_train_shard()

    def load_new_train_shard(self):
        # stream a fresh 500-row shard, advancing the offset each time
        train_rows, next_offset = get_tinystories_rows(
            "train", self._train_offset, self.rows_per_shard, self.max_context
        )
        if not train_rows:
            # wrapped past the end -> start over from the top
            self._train_offset = 0
            train_rows, next_offset = get_tinystories_rows(
                "train", 0, self.rows_per_shard, self.max_context
            )
        self._train_offset = next_offset
        self._shard_index += 1
        self.train_dataset = TinyStoriesDataset(train_rows, self.max_context)
        print(
            f"[data] loaded train shard #{self._shard_index}: "
            f"{len(train_rows)} stories -> {len(self.train_dataset)} examples "
            f"(next offset {self._train_offset})"
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=2,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=2,
        )
