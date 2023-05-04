from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.seed import * # seed setting module
from dataloader import *
from models import *


def main():
    # set seed
    seed = get_seed()
    set_seed(*seed)

    wandb_logger = WandbLogger(project="klue-re-001", name=f"seed:{'_'.join(map(str, seed))}")
    dataloader = Dataloader('klue/roberta-large', False, 32, 32, True, "~/dataset/train/train.csv",
                            "~/dataset/test/test_cheat.csv", "~/dataset/test/test_cheat.csv",
                            "~/dataset/test/test_data.csv")

    total_steps = (32470 // (12 * 4) + (32470 % (12 * 4) != 0)) * 5
    warmup_steps = int(0.1 * (32470 // (12 * 4) + (32470 % (12 * 4) != 0)))
    model = TypedEntityMarkerPuncModel(
        'klue/roberta-large',           # model name
        3e-5,                           # lr
        0.01,                           # weight decay
        "LDAM",                         # loss function
        warmup_steps,                   # warm up steps
        total_steps                     # total steps
    )

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        # fast_dev_run=True,                    # 검증용
        precision=16,                           # 16-bit mixed precision
        gpus = 1,
        accelerator='gpu',                      # GPU 사용
        # reload_dataloaders_every_n_epochs=1,  # dataloader를 매 epoch마다 reload해서 resampling
        accumulate_grad_batches=4,              # 4step만큼 합친 후 역전파
        max_epochs=5,                           # 최대 epoch 수
        logger=wandb_logger,                    # wandb logger 사용
        log_every_n_steps=1,                    # 1 step마다 로그 기록
        val_check_interval=0.5,                 # 0.25 epoch마다 validation
        callbacks=[

            LearningRateMonitor(logging_interval='step'), # learning rate를 매 step마다 기록
            EarlyStopping(                      # validation pearson이 8번 이상 개선되지 않으면 학습을 종료
                'val_f1',
                patience=8,
                mode='max',
                check_finite=False
            ),
            ModelCheckpoint(
                './save/',
                'klue_re_001_{val_f1:.4f}',
                monitor='val_f1',
                save_top_k=1,
                mode='max'
            )
        ]
    ) # yapf: disable

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)


if __name__ == "__main__":
    main()