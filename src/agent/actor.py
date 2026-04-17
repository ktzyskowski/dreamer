import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from src.nets.mlp import MultiLayerPerceptron
from src.util.functions import mixin_uniform


class DiscreteActor(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: DictConfig):
        super().__init__()
        self.action_size = output_size
        self.net = MultiLayerPerceptron(
            input_dim=input_size,
            hidden_dims=config.actor.hidden_dims,
            output_dim=output_size,
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
