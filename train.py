from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.seed import * # seed setting module
from utils.callbacks import *
from dataloader import *
from models import *
from config.config import config
import pandas as pd
import os


def main(seed):
    # set seed
    # seed = get_seed(seed)
    set_seed(seed)

    wandb_logger = WandbLogger(entity="line1029-academic-team",
                               project="query-change",
                               name=f"query-change-question:{seed}")
    dataloader = Dataloader(config.model_name, False, config.batch_size, config.batch_size, True,
                                            config.train_path, config.dev_path, config.test_path, config.predict_path)

    warmup_steps = total_steps = 0.
    if "warm_up_ratio" in config._asdict().keys():
        num_samples = pd.read_csv(config.train_path).shape[0]
        total_steps = (num_samples // (config.batch_size * 2) + (num_samples %
                                                                 (config.batch_size * 2) != 0)) * config.max_epoch
        warmup_steps = int(config.warm_up_ratio * (num_samples // (config.batch_size * 2) +
                                                   (num_samples % (config.batch_size * 2) != 0)))
    model = TypedEntityMarkerPuncModel(
        config.model_name,                           # model name
        config.learning_rate,                        # lr
        config.weight_decay,                         # weight decay
        config.loss_func,                            # loss function
        warmup_steps,                                # warm up steps
        total_steps,                                 # total steps
        # config.LDAM_start,
        lr_scheduler=config.lr_scheduler
        ) # yapf: disable

    ver = set_version()
    model_path = f"{get_time_str()}_{next(ver):0>4}"

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        # fast_dev_run=True,                    # 검증용
        precision=16,                           # 16-bit mixed precision
        gpus = 1,
        accelerator='gpu',                      # GPU 사용
        # reload_dataloaders_every_n_epochs=1,  # dataloader를 매 epoch마다 reload해서 resampling
        accumulate_grad_batches=2,              # 4step만큼 합친 후 역전파
        max_epochs=config.max_epoch,                           # 최대 epoch 수
        logger=wandb_logger,                    # wandb logger 사용
        log_every_n_steps=1,                    # 1 step마다 로그 기록
        val_check_interval=0.25,                 # 0.25 epoch마다 validation
        callbacks=[

            LearningRateMonitor(logging_interval='step'), # learning rate를 매 step마다 기록
            EarlyStopping(                      # validation pearson이 8번 이상 개선되지 않으면 학습을 종료
                'val_f1',
                patience=8,
                mode='max',
                check_finite=False
            ),
            CustomModelCheckpoint(
                './save/',
                model_path + '_{val_f1:.4f}',
                monitor='val_f1',
                save_top_k=1,
                mode='max'
            )
        ]
    ) # yapf: disable

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    path_dir = '/opt/ml/level2_klue-nlp-11/save'
    file_list = os.listdir(path_dir)
    for file in file_list:
        if file.startswith(model_path):
            path = os.path.join(path_dir, file)
            trainer.test(model=TypedEntityMarkerPuncModel.load_from_checkpoint(path), datamodule=dataloader)
            break
    wandb.finish()


if __name__ == "__main__":
    for i in range(3):
        main(i)