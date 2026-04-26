import torch
import torch.nn as nn


class ExpMovingAverage(nn.Module):
    """Exponential moving average transform module."""

    def __init__(self, decay: float):
        """Create new EMA transform.

        Args:
            decay (float): rate of decay of average.
        """
        super().__init__()
        self.decay = decay
        self.register_buffer("average", torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        """Transform the given input through the EMA.

        The average is updated according to the EMA definition, and returned.

        Args:
            x (torch.Tensor): the given input.
        Returns:
            average (torch.Tensor): the updated EMA value.
        """
        assert self.average.shape == x.shape
        self.average = (self.decay * self.average + (1 - self.decay) * x).detach()
        return self.average
