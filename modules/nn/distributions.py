import torch
from torch import nn

from modules.nn.mlp import MultiLayerPerceptron


# Used as:
# - Continue predictor
#     c_t ~ p(c_t | h_t, z_t)
class BernoulliModel(MultiLayerPerceptron):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation=nn.ReLU,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(self, x):
        """
        Args:
            model_state (batch, sequence, feature)
        Returns:
            distribution (torch.Bernoulli)
        """
        logits = super().forward(x)

        # (batch, sequence, 1) -> (batch, sequence)
        logits = logits.squeeze(-1)

        distribution = torch.distributions.Bernoulli(logits=logits)
        return distribution


# Used as:
# - Reward predictor
#     r_t ~ p(r_t | h_t, z_t)
# - Critic
#     v_t ~ p(v_t | h_t, z_t)
class NormalModel(MultiLayerPerceptron):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation=nn.ReLU,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,
            activation=activation,
        )

    def forward(self, x):
        """
        Args:
            model_state (batch, sequence, feature)
        Returns:
            distribution (torch.Normal)
        """
        output = super().forward(x)

        # (batch, sequence, 2) -> 2x(batch, sequence, 1)
        mean, log_std = output.chunk(2, dim=-1)

        # (batch, sequence, 1) -> (batch, sequence)
        mean = mean.squeeze(-1)

        log_std = log_std.squeeze(-1)
        distribution = torch.distributions.Normal(mean, torch.exp(log_std))
        return distribution
