import torch


def get_device(priority=None) -> str:
    """Get the torch device to use for training.

    Prioritizes the selected device, if available. Otherwise, will
    follow the given ranking (highest priority to leftmost device):

    `[priority]` >> `cuda` >> `mps` >> `cpu`
    """

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    # if user specified a device preference
    if priority:
        if priority in devices:
            selected_device = priority
        else:
            selected_device = devices[-1]
            logging.info(
                "Device '{}' not available, falling back to '{}'",
                priority,
                selected_device,
            )
        return selected_device

    # otherwise, return highest ranking available device
    return devices[-1]
