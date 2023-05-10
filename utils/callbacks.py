import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from datetime import datetime
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


# 모델 저장을 위한 class
class CustomModelCheckpoint(ModelCheckpoint):

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        monitor_candidates = self._monitor_candidates(trainer,
                                                      epoch=trainer.current_epoch,
                                                      step=trainer.global_step - 1)
        current = monitor_candidates.get(self.monitor)

        if (self._should_skip_saving_checkpoint(trainer) or self._save_on_train_epoch_end or self._every_n_epochs < 1 or
            (trainer.current_epoch + 1) % self._every_n_epochs != 0 or torch.isnan(current) or current < 0.6):
            return
        self.save_checkpoint(trainer)


def get_time_str():
    now = datetime.now()
    return now.strftime('%y-%m-%d-%H-%M')


# set version to save model
def set_version():
    for i in range(1, 1000):
        yield i


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)