import torch


def count_parameters(model):
    """Count the number of trainable parameters in a given model.

    Args:
        - model (nn.Module)
    Returns:
        - n_parameters
    """
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters


def deconstruct_batch(batch):
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    recurrent_states = batch["recurrent_states"]
    return observations, actions, rewards, dones, recurrent_states


def mixin_uniform(probs, split=0.01, dim=-1):
    # create uniform distribution over specified dimension
    uniform = torch.ones_like(probs) / probs.shape[dim]
    # mix uniform with given distribution
    mixed = (1 - split) * probs + split * uniform
    return mixed


def symlog(x):
    # Eq (9)
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    # Eq (9)
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
