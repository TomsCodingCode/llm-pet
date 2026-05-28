import torch
import pytorch_lightning as pl

from llm_pet.data_set.gsm8k.gsm8k_access import get_gsm8k_rows, GSM8K_SPLIT_SIZES, ROWS_PER_SHARD, VALIDATION_ROWS
from llm_pet.data_set.gsm8k.gsm8k_dataset import Gsm8kDataset, collate


class Gsm8kDataModule(pl.LightningDataModule):
    def __init__(self, rows_per_shard=ROWS_PER_SHARD, validation_rows=VALIDATION_ROWS,
                 batch_size=64, max_context=128, seed=42):
        super().__init__()
        self.rows_per_shard = rows_per_shard
        self.validation_rows = validation_rows
        self.batch_size = batch_size
        self.max_context = max_context

        # torch generator for picking the (fixed) validation offset
        self._gen = torch.Generator().manual_seed(seed)
        self._train_offset = 0        # advances as we stream through the corpus
        self.train_dataset = None
        self.val_dataset = None
        self._shard_index = 0

    def setup(self, stage=None):
        if self.val_dataset is None:
            # GSM8k has no validation split -> hold out a fixed slice of "test"
            # so val loss is comparable across shard swaps.
            high = max(1, GSM8K_SPLIT_SIZES["test"] - self.validation_rows)
            val_offset = int(torch.randint(0, high, (1,), generator=self._gen).item())
            val_rows, _ = get_gsm8k_rows(
                "test", val_offset, self.validation_rows, self.max_context)
            self.val_dataset = Gsm8kDataset(val_rows, self.max_context)
            print(f"[data] validation shard: {len(val_rows)} QA pairs "
                  f"-> {len(self.val_dataset)} examples")
        if self.train_dataset is None:
            self.load_new_train_shard()

    def load_new_train_shard(self):
        # stream a fresh shard from the train split, advancing the offset each time
        train_rows, next_offset = get_gsm8k_rows(
            "train", self._train_offset, self.rows_per_shard, self.max_context)
        if not train_rows:
            # wrapped past the end -> start over from the top
            self._train_offset = 0
            train_rows, next_offset = get_gsm8k_rows(
                "train", 0, self.rows_per_shard, self.max_context)
        self._train_offset = next_offset
        self._shard_index += 1
        self.train_dataset = Gsm8kDataset(train_rows, self.max_context)
        print(f"[data] loaded train shard #{self._shard_index}: "
              f"{len(train_rows)} QA pairs -> {len(self.train_dataset)} examples "
              f"(next offset {self._train_offset})")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate, num_workers=2, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate, num_workers=2)
