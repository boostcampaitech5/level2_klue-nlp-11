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
    # wandb_logger = WandbLogger(project="klue-re")
    dataloader = EntityVerbalizedDataloader('klue/roberta-large', False, 32, 32, True, "~/dataset/train/dummy.csv",
                                            "~/dataset/train/dummy.csv", "~/dataset/train/dummy.csv",
                                            "~/dataset/test/test_data.csv")
    if path.endswith(".ckpt"):
        model = TypedEntityMarkerPuncModel.load_from_checkpoint("./save/" + path)
    elif path.endswith(".pt"):
        model = torch.load("./save/" + path)

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
        'no_symbol_query_23-05-10-01-44_0001_val_f1=85.3087.pt',
        'no_symbol_query_23-05-10-04-21_0004_val_f1=85.5981.pt',
        'no_symbol_query_23-05-10-02-33_0002_val_f1=86.0387.pt',
        'no_symbol_query_23-05-10-05-07_0005_val_f1=85.6379.pt',
        'no_symbol_query_23-05-10-03-32_0003_val_f1=85.6158.pt', 'no_symbol_query_23-05-10-06-14_0006_val_f1=86.5456.pt'
    ]
    for path in save_path:
        main(path)
