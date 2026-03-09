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
    return observations, actions, rewards, dones
