import torch
from utils.utils import NanException
from utils.utils import noise

import numpy as np


if __name__ == "__main__":
    a = torch.LongTensor([0, 4])
    b = torch.Tensor([1, 2, 3, 4, 5, 6, 7])
    c = torch.index_select(b, 0, a)
    print(c)
