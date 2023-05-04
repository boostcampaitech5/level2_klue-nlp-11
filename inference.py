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
    dataloader = Dataloader('klue/roberta-large', False, 32, 32, True, "~/dataset/train/val.csv",
                            "~/dataset/train/val.csv", "~/dataset/train/val.csv", "~/dataset/test/test_data.csv")

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
    output.to_csv(f"output-{path}.csv", index=False)


if __name__ == "__main__":
    save_path = [
        'klue_re_0002_val_f1=69.1298.ckpt', 'klue_re_0012_val_f1=69.1136.ckpt', 'klue_re_0014_val_f1=69.7686.ckpt',
        'klue_re_0011_val_f1=69.0344.ckpt', 'klue_re_0013_val_f1=68.4460.ckpt', 'klue_re_0017_val_f1=67.9602.ckpt'
    ]
    for path in save_path:
        main(path)
