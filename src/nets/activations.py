import torch.nn as nn
import torch.nn.functional as F


class RMSNormSiLU(nn.Module):
    """RMSNorm + SiLU activation module."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.rms_norm(x, (x.shape[-1],))
        x = F.silu(x)
        return x
