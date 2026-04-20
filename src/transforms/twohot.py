import torch
import torch.nn as nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Compute symlog function.

    Eq (9) in paper.

    Args:
        x (*): input tensor.
    Returns:
        y (*): output tensor.
    """
    y = torch.sign(x) * torch.log(torch.abs(x) + 1)
    return y


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Compute symexp function.

    Eq (9) in paper.

    Args:
        x (*): input tensor.
    Returns:
        y (*): output tensor.
    """
    y = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return y


class SymlogTwoHot(nn.Module):
    """Symlog two-hot transform module."""

    def __init__(self, low: float, high: float, n_bins: int):
        super().__init__()
        self.low = low
        self.high = high
        self.n_bins = n_bins

        self.bins: torch.Tensor
        self.register_buffer("bins", torch.linspace(self.low, self.high, self.n_bins))

    def encode(self, y):
        """Encode the given tensor of scalar values into symlog two-hot encoding.

        Args:
            y (*): tensor of values, can be any shape.
        Returns:
            encoded (*, bins): tensor of symlog two-hot encodings.
        """
        y_symlog = symlog(y)

        # find lower bin indices
        k = torch.bucketize(y_symlog, self.bins) - 1
        # clamp bin indices ot [0, n_bins-2] to handle values that lie outside edges
        k = k.clamp(0, self.n_bins - 2)

        # bin weights (1.0 split between two bins: upper and lower)
        upper_weight = torch.abs(self.bins[k] - y_symlog) / torch.abs(
            self.bins[k + 1] - self.bins[k]
        )
        lower_weight = 1.0 - upper_weight

        # scatter weights into new tensor with added bin dimension: (*, bins)
        # unsqueeze() aligns dimensions: (*) -> (*, bins)
        encoded = torch.zeros(*y.shape, self.n_bins, device=y.device)
        encoded.scatter_(-1, k.unsqueeze(-1), lower_weight.unsqueeze(-1))
        encoded.scatter_(-1, (k + 1).unsqueeze(-1), upper_weight.unsqueeze(-1))

        return encoded

    def decode(self, encoded):
        """Decode the given tensor of symlog two-hot encoded values.

        This method accepts arbitrary tensor shapes, but the last dimension be `n_bins`.

        Args:
            encoded (*, bins): symlog two-hot encoded tensor.
        Returns:
            y (*): decoded tensor of scalar values.
        """
        weighted_bins = torch.softmax(encoded, dim=-1) * self.bins
        y_symlog = weighted_bins.sum(-1)
        y = symexp(y_symlog)
        return y
