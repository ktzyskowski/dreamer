import torch
import torch.nn.functional as F
from torch import nn

from modules.nn.mlp import MultiLayerPerceptron
from modules.nn.utils import mixin_uniform


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden_dims, action_dim):
        """
        Args:
            input_dim (int): model state dimension, recurrent + latent.
            action_dim (int): number of actions to select.
        """
        super().__init__()
        self.net = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
        )

    def forward(self, state):
        """
        Args:
            state (model_state): full model state, including recurrent and discrete state vectors.
        Returns:
            probs (actions): softmax distribution over discrete actions.
        """
        logits = self.net(state)
        probs = F.softmax(logits, -1)
        probs = mixin_uniform(probs, split=0.01, dim=-1)
        return probs

    def sample(self, state):
        probs = self(state)
        action = torch.multinomial(probs, 1).squeeze(-1)
        return action
