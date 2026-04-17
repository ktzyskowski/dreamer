import torch.nn as nn
import torch.nn.functional as F


class RMSNormSiLU(nn.Module):
    def __init__(self):
        super().__init__(self)

    def forward(self, x):
        x = F.rms_norm(x, x.shape[-1])
        x = F.silu(x)
        return x
