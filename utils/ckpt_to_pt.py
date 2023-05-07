import torch
from models import *
import os

if __name__ == "__main__":
    candidates_paths = [
        'klue_re_23-05-06-23-05_0006_val_f1=87.7206.ckpt', 'klue_re_23-05-06-16-59_0004_val_f1=88.8362.ckpt',
        'klue_re_23-05-06-21-15_0004_val_f1=88.1875.ckpt'
    ]

    for path in candidates_paths:
        # ckpt -> pt
        model = TypedEntityMarkerPuncModel.load_from_checkpoint("~/model_saves/" + path)
        save_path = os.path.expanduser("~/model_saves/" + path[:-4] + "pt")
        torch.save(model, save_path)