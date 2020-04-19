import torch
from torch import nn
from math import log2
from utils import leaky_relu

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(0.2),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(0.2)
        )

        self.downsample = nn.Conv2d(
            filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = x + res
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, transparent=False):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + \
            [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))

        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind < (len(chan_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample=is_not_last
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)
        self.to_logit = nn.Linear(2 * 2 * filters[-1], 1)

    def forward(self, x):
        b, *_ = x.shape
        x = self.blocks(x)
        x = x.reshape(b, -1)
        x = self.to_logit(x)
        return x.squeeze()
