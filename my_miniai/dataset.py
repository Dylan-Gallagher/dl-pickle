from __future__ import annotations
import math,numpy as np,matplotlib.pyplot as plt
from operator import itemgetter
from itertools import zip_longest
import fastcore.all as fc
from torch.utils.data import default_collate
from .minibatch import *

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Dataloader:
    def __init__(self, ds, bs):
        self.ds = ds
        self.bs = bs

    def __iter__(self):
        for i in range(0, len(self.ds), self.bs):
            yield self.ds[i:i+self.bs]


class Sampler:
    def __init__(self, ds, shuffle=False):
        self.n = len(ds)
        self.shuffle = shuffle

    def __iter__(self):
        res = list(range(self.n))
        if self.shuffle:
            random.shuffle(res)
        return iter(res)


class BatchSampler:
    def __init__(self, sampler, bs, drop_last=False):
        self.sampler = sampler
        self.bs = bs
        self.drop_last = drop_last

    def __iter__(self):
        yield from fc.chunked(iter(self.sampler), self.bs, drop_last=self.drop_last)


def collate(b):