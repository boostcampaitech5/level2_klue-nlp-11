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


def main(path):
    # wandb_logger = WandbLogger(project="klue-re")
    dataloader = EntityVerbalizedDataloader('klue/roberta-large', False, 12, 12, True, "~/dataset/train/val.csv",
                                            "~/dataset/train/val.csv", "~/dataset/train/val.csv",
                                            "~/dataset/test/test_data.csv")

    model = TypedEntityMarkerPuncModel.load_from_checkpoint("./save/" + path)

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
    output.to_csv(f"output-{path[:-5]}.csv", index=False)


if __name__ == "__main__":
    save_path = [
        'klue_re_23-05-05-06-39_0001_val_f1=70.7466.ckpt', 'klue_re_23-05-05-09-37_0002_val_f1=72.5724.ckpt',
        'klue_re_23-05-05-12-06_0004_val_f1=71.5972.ckpt', 'klue_re_23-05-05-08-08_0001_val_f1=70.1744.ckpt',
        'klue_re_23-05-05-10-51_0003_val_f1=70.8187.ckpt'
    ]
    for path in save_path:
        main(path)
