import pytorch_lightning as pl
import torch

from llm_pet.data_set.vocab import VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, encode, decode
from llm_pet.model.transformer import Transformer


class SLMLightning(pl.LightningModule):
    def __init__(
        self,
        num_layers=4,
        dim=128,
        num_heads=4,
        max_context=128,
        vocab_size=VOCAB_SIZE,
        p_drop=0.1,
        learning_rate=3e-4,
        max_steps=20000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            max_context=max_context,
            vocab_size=vocab_size,
            p_drop=p_drop,
        )
        self.learning_rate = learning_rate
        self.max_steps_total = max_steps

    def forward(self, dec_in, context, dec_in_pad_mask=None, context_pad_mask=None):
        return self.model(
            dec_in,
            context,
            x_pad_mask=dec_in_pad_mask,
            context_pad_mask=context_pad_mask,
        )

    def _step(self, batch):
        context, dec_in, target, context_pad_mask, dec_in_pad_mask = batch
        logits = self.model(
            dec_in,
            context,
            x_pad_mask=dec_in_pad_mask,
            context_pad_mask=context_pad_mask,
        )  # (B, T, V)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.long().reshape(-1),
            ignore_index=PAD_ID,
        )  # ignore padded targets
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # cosine decay to ~1/10th of the base LR over the full training run
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_steps_total, eta_min=self.learning_rate * 0.1
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    @torch.no_grad()
    def generate_next_sentence(self, context_text, max_tokens=None):
        # convenience helper: given a context string, sample the next sentence
        self.eval()
        max_tokens = max_tokens or self.hparams.max_context
        ctx_ids = encode(context_text)[-self.hparams.max_context :]
        context = torch.tensor([ctx_ids], device=self.device)
        out = torch.tensor([[BOS_ID]], device=self.device)

        out = self.model.generate(context, out, max_tokens=max_tokens)[0].tolist()
        generated = out[1:]  # drop the leading <bos>
        if EOS_ID in generated:
            generated = generated[: generated.index(EOS_ID)]
        return decode(generated)
