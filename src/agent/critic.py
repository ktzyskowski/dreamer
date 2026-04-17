import copy

from omegaconf import DictConfig
from torch import nn
import torch

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


class DualCritic(nn.Module):
    """Implements slow critic (critic target network).

    There are two critics, fast and slow. The fast critic outputs logits and is modified rapidly
    during gradient descent. The slow critic is an exponentially moving average of the fast critic,
    and is used to output actual critic values for during lambda return calculation. This provides
    a more stable signal for the actor/critic to chase (otherwise, there is a strong circular dependency).
    """

    def __init__(self, input_size: int, config: DictConfig):
        super().__init__()
        self.fast = Critic(input_size, config)
        self.slow = copy.deepcopy(self.fast)
        self.decay = config.critic.ema_decay

        for p in self.slow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_slow(self):
        for fast_p, slow_p in zip(self.fast.parameters(), self.slow.parameters()):
            new_p = self.decay * slow_p + (1 - self.decay) * fast_p
            slow_p.copy_(new_p)
