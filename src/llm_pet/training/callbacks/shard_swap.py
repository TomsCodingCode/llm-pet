import pytorch_lightning as pl


class ShardSwapCallback(pl.Callback):
    """Swap the training shard when the model starts overfitting it.

    Two complementary triggers, so we can swap even when losses are low:
      * absolute:  val_loss - train_loss > abs_gap        (early divergence)
      * relative:  train_loss / val_loss < ratio          (val >> train, low loss)

    On a swap we also notify the (optional) early-stopping callback so it grants
    the fresh shard a grace period before its val-loss regressions can count
    toward stopping. The swap callback must be listed BEFORE the early-stopping
    callback so the grace period is set before that round's stopping check runs.
    """

    def __init__(self, datamodule, abs_gap=0.30, ratio=0.80, early_stopping=None):
        super().__init__()
        self.datamodule = datamodule
        self.abs_gap = abs_gap
        self.ratio = ratio
        self.early_stopping = early_stopping  # GraceEarlyStopping or None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        val_loss = metrics.get("val_loss")
        if train_loss is None or val_loss is None:
            return

        t = float(train_loss)
        v = float(val_loss)
        gap = v - t
        ratio = t / v if v > 0 else 1.0
        absolute_trigger = gap > self.abs_gap
        relative_trigger = ratio < self.ratio

        print(
            f"[swap-check] step {trainer.global_step}: "
            f"train={t:.3f} val={v:.3f} gap={gap:.3f} t/v={ratio:.3f}"
        )

        if absolute_trigger or relative_trigger:
            why = "abs gap" if absolute_trigger else "relative t/v"
            print(f"[swap-check] {why} triggered -> swapping training shard")
            self.datamodule.load_new_train_shard()
            # force Lightning (2.x) to rebuild the train dataloader from the new
            # shard; reload_dataloaders_every_n_epochs=1 keeps boundaries frequent
            trainer.fit_loop._combined_loader = None
            # give the new shard a grace period before early stopping can react
            if self.early_stopping is not None:
                self.early_stopping.start_grace()
