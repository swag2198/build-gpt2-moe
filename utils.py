import os
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

    
def load_tokens(filename: str):
    # expects .npy file, used inside dataloader
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def save_checkpoint(model: nn.Module, optimizer=None, name='', root_dir='./checkpoint'):
    dic = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        dic["optimizer_state_dict"] = optimizer.state_dict()

    filename = os.path.join(root_dir, f'model_{name}.pth')
    torch.save(dic, filename)
