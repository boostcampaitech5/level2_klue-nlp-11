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
        'no_symbol_query_23-05-09-20-40_0008_val_f1=83.9765.pt',
        'no_symbol_query_23-05-09-16-49_0001_val_f1=85.5337.pt',
        'no_symbol_query_23-05-09-21-11_0009_val_f1=85.0460.pt',
        'no_symbol_query_23-05-09-17-20_0002_val_f1=85.9626.pt',
        'no_symbol_query_23-05-09-21-41_0010_val_f1=85.2004.pt',
        'no_symbol_query_23-05-09-17-51_0003_val_f1=85.8523.pt',
        'no_symbol_query_23-05-09-22-12_0011_val_f1=85.8519.pt',
        'no_symbol_query_23-05-09-18-23_0004_val_f1=85.9604.pt',
        'no_symbol_query_23-05-09-22-44_0012_val_f1=86.6466.pt',
        'no_symbol_query_23-05-09-18-58_0005_val_f1=85.0558.pt',
        'no_symbol_query_23-05-09-23-15_0013_val_f1=85.8766.pt',
        'no_symbol_query_23-05-09-19-32_0006_val_f1=85.1301.pt',
        'no_symbol_query_23-05-09-23-46_0014_val_f1=84.9482.pt',
        'no_symbol_query_23-05-09-20-08_0007_val_f1=84.8096.pt',
        'no_symbol_query_23-05-10-00-16_0015_val_f1=85.4417.pt',
    ]
    for path in save_path:
        main(path)
