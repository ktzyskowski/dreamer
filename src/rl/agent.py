from typing import Type

import torch.nn as nn

from src.rl.critic import DualCritic
from src.nets.mlp import MultiLayerPerceptron


class Agent(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        action_size: int,
        n_bins: int,
        activation: Type[nn.Module],
        critic_decay: float,
    ):
        super().__init__()

        self.actor = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_size,
            activation=activation,
            output_activation=None,
        )
        self.critic = DualCritic(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=n_bins,
            activation=activation,
            decay=critic_decay,
        )
