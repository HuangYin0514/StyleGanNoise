import sys
sys.path.append("..") # 这句是为了导入_config
import torch
import numpy as np
from utils import noise


if __name__ == "__main__":
    a = noise(3, 512)
    print(a)
