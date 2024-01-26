from  import *

m = 28 * 28
nh = 50


def get_model():
    return nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, 10))

