import torch.nn as nn


def flatten(d: dict, sep="."):
    """Flatten the given dictionary."""

    def flatten_helper(d: dict, prefix: str = ""):
        items = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(flatten_helper(v, key))
            else:
                items[key] = v
        return items

    return flatten_helper(d)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a given model."""
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters
