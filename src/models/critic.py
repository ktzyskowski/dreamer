from omegaconf import DictConfig
from torch import nn

from src.nets.mlp import MultiLayerPerceptron
from src.util.two_hot import TwoHot


class Critic(nn.Module):
    def __init__(self, input_size: int, config: DictConfig):
        super().__init__()
        self.net = MultiLayerPerceptron(
            input_dim=input_size,
            hidden_dims=config.critic.hidden_dims,
            output_dim=config.two_hot.n_bins,
        )

        # initialize weights of output layer to be zero,
        # done to avoid hallucinating rewards early in training
        nn.init.zeros_(self.net.net[-1].weight)  # type: ignore
        nn.init.zeros_(self.net.net[-1].bias)  # type: ignore

    def forward(self, state):
        """Returns both the logits (for loss) and decoded scalar value."""
        logits = self.net(state)
        return logits
