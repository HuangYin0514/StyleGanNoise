import os
import sys
import math
import fire
import json
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
import torch.backends.cudnn as cudnn

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch_optimizer import DiffGrad
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from PIL import Image
from pathlib import Path
from utils.utils import *



class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu(0.2)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

