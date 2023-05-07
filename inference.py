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
    output.to_csv(f"{path[:-5]}.csv", index=False)


if __name__ == "__main__":
    save_path = [
        'klue_re_23-05-07-03-31_0011_val_f1=88.3111.ckpt', 'klue_re_23-05-06-20-06_0003_val_f1=88.2921.ckpt',
        'klue_re_23-05-06-23-05_0006_val_f1=87.7206.ckpt', 'klue_re_23-05-07-01-36_0009_val_f1=88.9049.ckpt',
        'klue_re_23-05-06-21-15_0004_val_f1=88.1875.ckpt', 'klue_re_23-05-06-23-45_0007_val_f1=88.5520.ckpt',
        'klue_re_23-05-07-02-32_0010_val_f1=87.9266.ckpt', 'klue_re_23-05-06-22-09_0005_val_f1=88.0122.ckpt',
        'klue_re_23-05-07-00-44_0008_val_f1=87.6694.ckpt'
    ]
    for path in save_path:
        main(path)
