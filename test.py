import torch
import numpy as np
# from run import *
import time
from retry.api import retry_call
from utils import *
import fire


def make_trouble(a=1, b=2, c=3, d=4, *args, **kwargs):
    print(a, b, c, d, d, e)
    print(*args)
    print(kwargs)


if __name__ == '__main__':
    total_noise_loss = torch.tensor(0.).to(device)
    if any(torch.isnan(l) for l in (total_noise_loss)):
        print(
            f'NaN detected for generator or discriminator. Loading from checkpoint '
        )
