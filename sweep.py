from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.seed import * # seed setting module
from utils.callbacks import *
from dataloader import *
from models import *
import wandb


def main():
    # HP Tuning
    # Sweep을 통해 실행될 학습 코드 작성
    sweep_config = {
        'method': 'random',                        # random: 임의의 값의 parameter 세트를 선택
        'parameters': {
            'learning_rate': {
                'values': [1e-5, 2e-5, 3e-5, 5e-5]
            },
            'max_epoch': {
                'values': [5]
            },
            'batch_size': {
                'values': [16, 24, 32]
            },
            'model_name': {
                'values': [
                    'klue/roberta-large',
                    # 'monologg/koelectra-base-v3-discriminator',
                    # 'beomi/KcELECTRA-base',
                    # 'rurupang/roberta-base-finetuned-sts',
                    # 'snunlp/KR-ELECTRA-discriminator'
                ]
            },
            'warm_up_ratio': {
                'values': [0, 0.1, 0.3, 0.6]
            },
            'weight_decay': {
                'values': [0, 0.01]
            },
            'loss_func': {
                'values': ["CE", "CB"]
            },
            # 'LDAM_start': {
            #     'values': [250, 500, 1000]
            # }
        },
        'metric': {
            'name': 'val_f1',
            'goal': 'maximize'
        }
    } # yapf: disable

    ver = set_version()

    def sweep_train(config=None):
        """
        sweep에서 config로 run
        wandb에 로깅

        Args:
            config (_type_, optional): _description_. Defaults to None.
        """

        with wandb.init(config=config) as run:
            config = wandb.config
            # set seed
            seed = get_seed()
            set_seed(*seed)
            run.name = f"seed:{'_'.join(map(str,seed))}"

            wandb_logger = WandbLogger(project="klue-re-sweep")
            dataloader = EntityVerbalizedDataloader(config.model_name, False, config.batch_size, config.batch_size,
                                                    True, "~/dataset/train/train_split.csv", "~/dataset/train/val.csv",
                                                    "~/dataset/train/val.csv", "~/dataset/test/test_data.csv")
            warmup_steps = total_steps = None
            if "warm_up_ratio" in config and config.warm_up_ratio:
                total_steps = (32470 // (12 * 4) + (32470 % (12 * 4) != 0)) * 5
                warmup_steps = int(config.warm_up_ratio * (32470 // (12 * 4) + (32470 % (12 * 4) != 0)))
            model = TypedEntityMarkerPuncModel(
                config.model_name,                           # model name
                config.learning_rate,                        # lr
                config.weight_decay,                         # weight decay
                config.loss_func,                            # loss function
                warmup_steps,                                # warm up steps
                total_steps,                                 # total steps
                # config.LDAM_start
                ) # yapf: disable
            # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
            trainer = pl.Trainer(
                # fast_dev_run=True,                    # 검증용
                precision=16,                           # 16-bit mixed precision
                gpus = 1,
                accelerator='gpu',                      # GPU 사용
                # # dataloader를 매 epoch마다 reload해서 resampling
                # reload_dataloaders_every_n_epochs=1,
                accumulate_grad_batches=2,              # 4step만큼 합친 후 역전파
                max_epochs=config.max_epoch,                           # 최대 epoch 수
                logger=wandb_logger,                    # wandb logger 사용
                log_every_n_steps=1,                    # 1 step마다 로그 기록
                val_check_interval=0.25,                # 0.25 epoch마다 validation
                callbacks=[
                    # learning rate를 매 step마다 기록
                    LearningRateMonitor(logging_interval='step'),
                    EarlyStopping(                      # validation pearson이 8번 이상 개선되지 않으면 학습을 종료
                        'val_f1',
                        patience=8,
                        mode='max',
                        check_finite=False
                    ),
                    CustomModelCheckpoint(
                        './save/',
                        f'klue_re_{get_time_str()}_{next(ver):0>4}_{{val_f1:.4f}}',
                        monitor='val_f1',
                        save_top_k=1,
                        mode='max'
                    )
                ]
            ) # yapf: disable
            trainer.fit(model=model, datamodule=dataloader)  # 모델 학습
            trainer.test(model=model, datamodule=dataloader) # 모델 평가

    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=sweep_config,              # config 딕셔너리 추가,
        entity="line1029-academic-team", # 팀 이름
        project="klue-re-sweep-001"      # project의 이름 추가
    )
    wandb.agent(
        sweep_id=sweep_id,               # sweep의 정보를 입력
        function=sweep_train,            # train이라는 모델을 학습하는 코드를
        count=80                         # 총 n회 실행
    )


if __name__ == "__main__":
    main()