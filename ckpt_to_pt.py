import torch
from models import *
import os

if __name__ == "__main__":
    candidates_paths = [
        'klue_re_23-05-07-06-29_0014_val_f1=88.2252.ckpt',
        'klue_re_23-05-07-17-33_0026_val_f1=88.8911.ckpt',
    ]

    for path in candidates_paths:
        # ckpt -> pt
        model = TypedEntityMarkerPuncModel.load_from_checkpoint("~/model_saves/" + path)
        save_path = os.path.expanduser("~/model_saves/" + path[:-4] + "pt")
        torch.save(model, save_path)