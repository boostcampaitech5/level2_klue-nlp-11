from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.seed import * # seed setting module
from dataloader import *
from models import *
import torch.nn.functional as F


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("./utils/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def main():
    # wandb_logger = WandbLogger(project="klue-re")
    dataloader = Dataloader('klue/roberta-large', 12, 12, True, "~/dataset/train/train.csv",
                            "~/dataset/train/train.csv", "~/dataset/test/test_cheat.csv",
                            "~/dataset/test/test_data.csv")

    total_steps = warmup_steps = None
    # model = BaseModel(
    #     'klue/roberta-large', # model name
    #     1e-5,                 # lr
    #     0.01,                 # weight decay
    #     "CB",                 # loss function
    #     None,                 # warm up steps
    #     None                  # total steps
    # )
    model = BaseModel.load_from_checkpoint("./save/klue_re_001_val_f1=71.1218.ckpt")

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        gpus = 1,
        accelerator='gpu'
    ) # yapf: disable


    # trainer.test(model=model, datamodule=dataloader)
    predictions_prob = torch.cat(trainer.predict(model=model, datamodule=dataloader))
    predictions_prob = F.softmax(predictions_prob, -1)
    predictions_label = predictions_prob.argmax(-1).tolist()
    predictions_prob = predictions_prob.tolist()
    predictions = num_to_label(predictions_label)

    output = pd.read_csv("~/dataset/sample_submission.csv")
    output["pred_label"] = predictions
    output["probs"] = predictions_prob
    output.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()