import copy
from typing import Type

from torch import nn
import torch

from src.nets.mlp import MultiLayerPerceptron


class DualCritic(nn.Module):
    """Dual critic network.

    There are two critics, fast and slow. The fast critic is modified rapidly during gradient descent.
    The slow critic is an exponentially moving average of the fast critic, and is used to output critic
    values for lambda return calculation. This provides a more stable signal for the actor/critic to chase
    (otherwise, if we only relied on the fast critic, the target would be too unstable).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: Type[nn.Module],
        decay: float,
    ):
        super().__init__()
        self.fast = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            output_activation=None,
        )

        # initialize last layer to output zero, to avoid hallucinating early rewards
        nn.init.zeros_(self.fast.net[-1].weight)  # type: ignore
        nn.init.zeros_(self.fast.net[-1].bias)  # type: ignore

        # slow network is a copy of the fast network
        self.slow = copy.deepcopy(self.fast)
        for param in self.slow.parameters():
            param.requires_grad_(False)

        self.decay = decay

    @torch.no_grad()
    def update_slow(self):
        """Update the parameters of the slow critic network."""
        for fast_param, slow_param in zip(
            self.fast.parameters(), self.slow.parameters()
        ):
            slow_param.lerp_(fast_param, 1 - self.decay)
