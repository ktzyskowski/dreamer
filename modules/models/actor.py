from torch import nn

from modules.nn.mlp import MultiLayerPerceptron


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        """
        Args:
            input_dim (int): model state dimension, recurrent + latent.
            action_dim (int): number of actions to select.
        """
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=[128, 128, 128],
            output_dim=action_dim * 2,
        )

    def forward(self, model_state, no_grad=False):
        action_params = self.mlp(model_state)
