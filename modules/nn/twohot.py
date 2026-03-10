import torch


class TwoHot:
    """Two-hot representation of continuous values."""

    def __init__(self, low: float, high: float, n_bins: int):
        self.low = low
        self.high = high
        self.n_bins = n_bins

    def encode(self, y):
        """
        Encode the given tensor of scalars into two-hot encoding.

        This method accepts any shape, and will add a new dimension for the bin values.

        Args:
            y (*): the tensor of scalars.
        Returns:
            twohot (*, bins): the tensor of two-hot vectors.
        """
        bins = torch.linspace(self.low, self.high, self.n_bins, device=y.device)

        # not necessary with symlog transformed inputs, but clamping helps guard against edge cases
        # y = y.clamp(self.low, self.high)

        # find lower bin indices
        k = torch.bucketize(y, bins) - 1
        # clamp bin indices to [0, n_bins-2] to handle values which lie beyond edges
        k = k.clamp(0, self.n_bins - 2)

        # interpolation weight for the upper bin
        upper_weight = torch.abs(bins[k] - y) / torch.abs(bins[k + 1] - bins[k])
        lower_weight = 1.0 - upper_weight

        # scatter weights into one-hot-shaped tensor: (*, bins)
        # unsqueezes align dimensions: (*) -> (*, bins)
        twohot = torch.zeros(*y.shape, self.n_bins, device=y.device)
        twohot.scatter_(-1, k.unsqueeze(-1), lower_weight.unsqueeze(-1))
        twohot.scatter_(-1, (k + 1).unsqueeze(-1), upper_weight.unsqueeze(-1))

        return twohot

    def decode(self, logits):
        """
        Decode the given tensor of logits back into scalar values.

        This method accepts arbitrary tensor shapes, but the last dimension be `n_bins`.

        Args:
            logits (*, bins): logits tensor.
        Returns:
            y (*): decoded scalar values tensor.
        """
        bins = torch.linspace(self.low, self.high, self.n_bins, device=logits.device)
        weighted_logits = torch.softmax(logits, dim=-1) * bins
        y = weighted_logits.sum(-1)
        return y
