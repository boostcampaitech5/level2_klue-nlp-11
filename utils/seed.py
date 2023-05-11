import random
import torch
import pytorch_lightning as pl
import numpy as np
import random

# get random seeds
def get_seed():
    return [random.randint(0, 2**32 - 1) for _ in range(6)]

# set seeds
def set_seed(a):
    torch.manual_seed(a)
    torch.cuda.manual_seed(a)
    torch.cuda.manual_seed_all(a) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(a)
    random.seed(a)
    pl.seed_everything(a, workers=True)