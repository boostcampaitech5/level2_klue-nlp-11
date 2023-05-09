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
                                                           # 'klue_re_23-05-07-12-47_0021_val_f1=88.7511.ckpt',
                                                           # 'klue_re_23-05-07-17-33_0026_val_f1=88.8911.ckpt',
                                                           # 'klue_re_23-05-07-22-18_0031_val_f1=87.7723.ckpt',
                                                           # 'klue_re_23-05-07-13-56_0022_val_f1=88.5217.ckpt',
        'klue_re_23-05-07-18-32_0027_val_f1=87.5601.ckpt',
        'klue_re_23-05-07-23-00_0032_val_f1=88.1676.ckpt',
        'klue_re_23-05-07-14-52_0023_val_f1=87.8127.ckpt',
        'klue_re_23-05-07-19-15_0028_val_f1=88.5585.ckpt',
        'klue_re_23-05-07-23-55_0033_val_f1=88.7061.ckpt',
        'klue_re_23-05-07-15-42_0024_val_f1=88.3356.ckpt',
        'klue_re_23-05-07-20-15_0029_val_f1=88.5262.ckpt',
        'klue_re_23-05-08-00-52_0034_val_f1=88.6500.ckpt',
        'klue_re_23-05-07-16-38_0025_val_f1=87.9711.ckpt',
        'klue_re_23-05-07-21-14_0030_val_f1=87.8359.ckpt',
    ]
    for path in save_path:
        main(path)
