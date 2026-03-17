from torch import nn
import torch


class ReinforceLoss(nn.Module):
    def __init__(self, eta=3e-4, lamda=0.97):
        super().__init__()

        self.eta = eta

        advantage_scale = torch.tensor(0, dtype=torch.float32)
        self.register_buffer("advantage_scale", advantage_scale)

    def forward(self, batch):
        pass
