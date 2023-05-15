import random
import torch
import pytorch_lightning as pl
import numpy as np
import random


# get random seeds
def get_seed():
    return [random.randint(0, 2**32 - 1) for _ in range(6)]


# set seeds
def set_seed(a, is_random=True):
    """_summary_

    Args:
        a (int or list): is_random==True인 경우 list, False인 경우 int
        is_random (bool, optional): 랜덤시드를 쓸지 고정시드를 쓸지에 대한 여부. Defaults to True.
    """
    if is_random:
        a, b, c, d, e, f = a
        torch.manual_seed(a)
        torch.cuda.manual_seed(b)
        torch.cuda.manual_seed_all(c) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(d)
        random.seed(e)
        pl.seed_everything(f, workers=True)
    else:
        torch.manual_seed(a)
        torch.cuda.manual_seed(a)
        torch.cuda.manual_seed_all(a) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(a)
        random.seed(a)
        pl.seed_everything(a, workers=True)
