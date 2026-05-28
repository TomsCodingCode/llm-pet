import pytorch_lightning as pl


class GraceEarlyStopping(pl.callbacks.EarlyStopping):
    """EarlyStopping that ignores val-loss regressions for `grace_period`
    validation checks after each shard swap.

    A swap changes the training distribution, so val_loss predictably bumps for
    a few checks afterwards. Without this, early stopping could fire on that
    transient. During the grace window we skip the stopping check entirely and
    keep the patience counter (`wait_count`) reset, so the fresh shard always
    gets a fair number of validations to settle before it can be stopped on.
    """

    def __init__(self, *args, grace_period=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.grace_period = grace_period
        self._grace_remaining = 0

    def start_grace(self):
        # called by the swap callback when a new shard is loaded
        self._grace_remaining = self.grace_period

    def _run_early_stopping_check(self, trainer):
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            self.wait_count = 0  # don't accumulate patience during grace
            # still track the best score so post-grace comparisons are sensible
            logs = trainer.callback_metrics
            current = logs.get(self.monitor)
            if current is not None and self.monitor_op(
                current.squeeze(), self.best_score.to(current.device)
            ):
                self.best_score = current.squeeze()
            print(
                f"[early-stop] grace period active "
                f"({self._grace_remaining} checks left) -> not stopping"
            )
            return
        super()._run_early_stopping_check(trainer)
