import torch
from minibatch import *

preds = torch.tensor([-0.09, -0.21, -0.08,  0.10, -0.04,  0.08, -0.04, -0.03,  0.01,  0.06])
yb = torch.tensor([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

print((preds.argmax()==yb).float().mean())