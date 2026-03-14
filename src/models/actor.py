import torch
import torch.nn.functional as F
from torch import nn

from modules.nn.mlp import MultiLayerPerceptron
from modules.nn.functions import mixin_uniform


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden_dims, action_dim):
        """
        Args:
            input_dim (int): model state dimension, recurrent + latent.
            action_dim (int): number of actions to select.
        """
        super().__init__()
        self.action_size = action_dim
        self.net = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
        )

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def forward(self, state):
        """Calculate action probabilities, and sample an action, from the given full model state."""
        logits = self.net(state)
        probs = F.softmax(logits, -1)
        probs = mixin_uniform(probs, split=0.01, dim=-1)
        action = torch.distributions.OneHotCategorical(probs=probs).sample()
        return action, probs
