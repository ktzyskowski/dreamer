import os
from typing import Any

import torch
import torch.optim as optim
import torch.nn as nn


class CheckpointManager:
    """Checkpoint manager class.

    Saves and loads training state, including models and step counters.
    Values in `modules` may be any state-dict provider (nn.Module, Optimizer).
    """

    def __init__(
        self,
        directory: str,
        modules: dict[str, nn.Module | optim.Optimizer],
        save_every_n_gradient_steps: int = 1_000,
    ):
        self.directory = directory
        self.modules = modules
        self.save_every_n_gradient_steps = save_every_n_gradient_steps

        # create checkpoint directory if doesn't exist
        os.makedirs(self.directory, exist_ok=True)

    def maybe_save(self, step: int, gradient_step: int):
        if step == 0 or gradient_step == 0:
            # don't save model before any training is done
            return

        if gradient_step % self.save_every_n_gradient_steps == 0:
            path = os.path.join(self.directory, f"checkpoint_{gradient_step:06d}.pt")
            self.save(path, step, gradient_step)

    def save(self, path: str, step: int, gradient_step: int):
        payload = {
            **{key: module.state_dict() for key, module in self.modules.items()},
            "step": step,
            "gradient_step": gradient_step,
        }
        torch.save(payload, path)

    def load(self, path: str, device: str) -> dict:
        checkpoint = torch.load(path, map_location=device)
        for key, module in self.modules.items():
            module.load_state_dict(checkpoint[key])
        return {
            "step": checkpoint["step"],
            "gradient_step": checkpoint["gradient_step"],
        }
