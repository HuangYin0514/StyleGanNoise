import torch
import numpy as np
# from run import *
import time
from retry.api import retry_call
from utils import *

a = 1


def make_trouble():
    global a
    a += 1
    print(a)
    raise NanException


if __name__ == '__main__':
    list1 = [1, 2, 3, 4]
    loader = cycle(list1)
    for _ in range(10):
        print(next(loader))
