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


def main(path: str):
    # for dev data
    # wandb_logger = WandbLogger(project="klue-re")

    dataloader = Dataloader("klue/roberta-large", False, 32, 32, True, "~/dataset/train/dummy.csv",
                            "~/dataset/train/dummy.csv", "~/dataset/train/dummy.csv", "~/dataset/test/test_data.csv")
    if path.endswith(".ckpt"):
        model = TypedEntityMarkerPuncModel.load_from_checkpoint("./save/" + path)
        end_idx = -5
    elif path.endswith(".pt"):
        model = torch.load("./save/" + path)
        end_idx = -3

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        gpus = 1,
        accelerator='gpu'
    ) # yapf: disable

    # for dev data
    # trainer.test(model=model, datamodule=dataloader)

    predictions_prob = torch.cat(trainer.predict(model=model, datamodule=dataloader))
    predictions_prob = F.softmax(predictions_prob, -1)
    predictions_label = predictions_prob.argmax(-1).tolist()
    predictions_prob = predictions_prob.tolist()
    predictions = num_to_label(predictions_label)

    output = pd.read_csv("~/dataset/sample_submission.csv")
    output["pred_label"] = predictions
    output["probs"] = predictions_prob
    output.to_csv(f"{path[:end_idx]}.csv", index=False)


if __name__ == "__main__":
    save_path = ['tmp_path']
    for path in save_path:
        main(path)
