import torch
from torch import nn

from src.util.ema import ExpMovingAverage


class CriticLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.target = ExpMovingAverage(decay=0.99)

    def forward(self, batch):
        pass
