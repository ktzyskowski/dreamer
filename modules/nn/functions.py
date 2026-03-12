import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a given model.

    Args:
        - model (nn.Module)
    Returns:
        - n_parameters: the total count of trainable model parameters.
    """
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters


# TODO: do I need this function?
def deconstruct_batch(batch):
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    recurrent_states = batch["recurrent_states"]
    return observations, actions, rewards, dones, recurrent_states


def mixin_uniform(probs: torch.Tensor, split=0.01, dim=-1) -> torch.Tensor:
    """Mixes a uniform distribution with the given probabilities.

    For a probability distribution with N outcomes, the mixed distribution
    will have probabilities:

    p_i = (1 - split)*p_i + (split)*1/N

    Args:
        probs (*): the tensor of probabilities.
        split (float): percentage assigned to . Defaults to 0.01.
        dim (int): tensor probability dimension. Defaults to -1, i.e. last dimension.

    Returns:
        torch.Tensor: mixed probability distribution.
    """
    # create uniform distribution over specified dimension
    uniform = torch.ones_like(probs) / probs.shape[dim]
    # mix uniform with given distribution
    mixed = (1 - split) * probs + split * uniform
    return mixed


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Compute symlog function.

    Eq (9) in paper.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Compute sympexp function.

    Eq (9) in paper.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
