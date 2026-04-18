import logging

import torch
import torch.nn as nn


def get_device(priority=None) -> str:
    """Get the torch device to use for training.

    Prioritizes the selected device, if available. Otherwise, will
    follow the given ranking (highest priority to leftmost device):

    `[priority]` >> `cuda` >> `mps` >> `cpu`
    """

    # get all available devices
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    if priority and priority in devices:
        # if user specified a device preference, use it
        selected_device = priority
    else:
        # otherwise, return highest ranking available device
        selected_device = devices[-1]
        logging.info(
            "Device '{}' not available, falling back to '{}'",
            priority,
            selected_device,
        )

    return selected_device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a given model."""
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters
