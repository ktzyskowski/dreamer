import torch


def mixin_uniform(probs: torch.Tensor, split=0.01, dim=-1) -> torch.Tensor:
    """Mixes a uniform distribution with the given probabilities.

    For a probability distribution with N outcomes, the mixed distribution
    will have probabilities:

    p_i = (1 - split)*p_i + (split)*1/N

    Args:
        probs (*): the tensor of probabilities.
        split (float): percentage assigned to uniform distribution. Defaults to 0.01.
        dim (int): tensor probability dimension. Defaults to -1, i.e. last dimension.

    Returns:
        torch.Tensor: mixed probability distribution.
    """
    # create uniform distribution over specified dimension
    uniform = torch.ones_like(probs) / probs.shape[dim]
    # mix uniform with given distribution
    mixed = (1 - split) * probs + split * uniform
    return mixed
