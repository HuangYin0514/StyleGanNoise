import torch
import numpy as np
from model import NoiseVectorizer


if __name__ == "__main__":
    noiseVectorizer = NoiseVectorizer(emb=100, depth=5)
    input = torch.FloatTensor(28, 100).uniform_(0., 1.)
    result = noiseVectorizer(input)
    print(result)
    print(result.shape)
