import torch
from pytorch_lightning import ModelCheckpoint
import pytorch_lightning as pl
from datetime import datetime


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
    return now.strftime('%y-%m-%d-%H:%M')

# set version to save model
def set_version():
    for i in range(1, 1000):
        yield i