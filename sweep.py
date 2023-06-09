from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.seed import * # seed setting module
from utils.utils import *
from dataloader import *
from models import *
import wandb
import os
import yaml


def main(is_random, experiment_name, experiment_idx):
    # HP Tuning
    # Sweep을 통해 실행될 학습 코드 작성
    with open("./config/sweep_config.yaml", "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

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
            seed_idx = next(ver)
            # set seed
            if is_random:
                seed = get_seed()
                set_seed(seed)
                run.name = f"{experiment_name}_seed:{'_'.join(map(str,seed))}"
            else:
                set_seed(seed_idx, False)
                run.name = f"{experiment_name}_seed:{seed_idx}"

            wandb_logger = WandbLogger(project=f"{experiment_name}-{experiment_idx:03}")
            dataloader = Dataloader(config.model_name, False, config.batch_size, config.batch_size, True,
                                    sweep_config["file_path"]["train_path"], sweep_config["file_path"]["dev_path"],
                                    sweep_config["file_path"]["test_path"], sweep_config["file_path"]["predict_path"])
            warmup_steps = total_steps = 0.
            if "warm_up_ratio" in config:
                num_samples = pd.read_csv(sweep_config["file_path"]["train_path"]).shape[0]
                total_steps = (num_samples // (config.batch_size * 2) +
                               (num_samples % (config.batch_size * 2) != 0)) * config.max_epoch
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

            model_path = f"{experiment_name}_{get_time_str()}_{seed_idx:0>4}"

            # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
            trainer = pl.Trainer(
                # fast_dev_run=True,                    # 검증용
                precision=16,                           # 16-bit mixed precision
                gpus = 1,
                accelerator='gpu',                      # GPU 사용
                # # dataloader를 매 epoch마다 reload해서 resampling
                # reload_dataloaders_every_n_epochs=1,
                accumulate_grad_batches=2,              # 2step만큼 합친 후 역전파
                max_epochs=config.max_epoch,                           # 최대 epoch 수
                logger=wandb_logger,                    # wandb logger 사용
                log_every_n_steps=1,                    # 1 step마다 로그 기록
                val_check_interval=0.25,                # 0.25 epoch마다 validation
                callbacks=[
                    # learning rate를 매 step마다 기록
                    LearningRateMonitor(logging_interval='step'),
                    EarlyStopping(                      # validation f1이 8번 이상 개선되지 않으면 학습을 종료
                        'val_f1',
                        patience=8,
                        mode='max',
                        check_finite=False
                    ),
                    CustomModelCheckpoint(
                        './save/',
                        model_path+'_{val_f1:.4f}',
                        monitor='val_f1',
                        save_top_k=1,
                        mode='max'
                    )
                ]
            ) # yapf: disable
            trainer.fit(model=model, datamodule=dataloader) # 모델 학습
            path_dir = '/opt/ml/level2_klue-nlp-11/save'
            file_list = os.listdir(path_dir)
            for file in file_list:
                if file.startswith(model_path) and file.endswith(".ckpt"):
                    path = os.path.join(path_dir, file)
                    model = TypedEntityMarkerPuncModel.load_from_checkpoint(path)
                    save_path = os.path.expanduser(path[:-4] + "pt")
                    torch.save(model, save_path)
                    if os.path.isfile(path):
                        os.remove(path)
                                                            # val_f1.max와 test_f1이 같은지 체크
                                                            # trainer.test(model=model, datamodule=dataloader)
                    break

    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=sweep_config,                              # config 딕셔너리 추가,
        entity="line1029-academic-team",                 # 팀 이름
        project=f"{experiment_name}-{experiment_idx:03}" # project의 이름 추가
    )
    wandb.agent(
        sweep_id=sweep_id,                               # sweep의 정보를 입력
        function=sweep_train,                            # train이라는 모델을 학습하는 코드를
        count=30                                         # 총 n회 실행
    )


if __name__ == "__main__":
    is_random = True
    experiment_name = "Test"
    experiment_idx = 1
    main(is_random, experiment_name, experiment_idx)