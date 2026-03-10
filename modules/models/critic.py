from torch import nn

from modules.nn.mlp import MultiLayerPerceptron
from modules.nn.twohot import TwoHot


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, low, high, n_bins):
        super().__init__()
        self.net = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=n_bins,
        )
        self.twohot = TwoHot(low, high, n_bins)

    def forward(self, model_state, return_logits=False):
        pass
