import torch
from torch import nn

from src.util.ema import ExpMovingAverage


class ActorLoss(nn.Module):
    def __init__(self, eta=3e-4):
        super().__init__()

        # entropy coefficient
        self.eta = eta

        self.advantage_scale = ExpMovingAverage(decay=0.99)

    def forward(self, batch):
        pass
