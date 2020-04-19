from torch import nn
from utils import leaky_relu


class NoiseVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb, emb),
            leaky_relu(0.2),
            nn.Linear(emb, 512),
            leaky_relu(0.2),
            nn.Linear(512, 1024),
            leaky_relu(0.2),
            nn.Linear(1024, 2048),
            leaky_relu(0.2),
            nn.Linear(2048, 4096),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).reshape(-1, 64, 64, 1)
