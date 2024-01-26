import torch
from torch import nn
from torch.utils.data import default_collate
from typing import Mapping
from minibatch import *
from dataset import *


def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def conv(num_inputs, num_filters, kernel_size=3, stride=2, act=True):
    result = nn.Conv2d(num_inputs, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
    if act:
        result = nn.Sequential(result, nn.ReLU())
    return result


def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)


def collate_device(b):
    return to_device(default_collate(b))
