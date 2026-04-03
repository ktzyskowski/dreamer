from torch import nn
import torch


class ExpMovingAverage(nn.Module):
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        self.register_buffer("average", torch.tensor(0.0))

    def forward(self, x):
        self.average = self.decay * self.average + (1 - self.decay) * x
        return self.average
