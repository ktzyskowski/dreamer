from torch import nn
import torch

from modules.nn.mlp import MultiLayerPerceptron

# full_state = cat[recurrent_state, latent_state]
# latent_posterior.shape == latent_prior.shape


# posterior = posterior_net(recurrent, encoded)
# prior = prior_net(recurrent)


class WorldModel(nn.Module):
    def __init__(
        self,
        observation_shape,
        recurrent_dim,
        latent_dim,
    ):
        super().__init__()

        self.recurrent_dim = recurrent_dim
        self.latent_dim = latent_dim
        self.model_state_dim = recurrent_dim + latent_dim

        self.reward_predictor = MultiLayerPerceptron(
            input_dim=self.model_state_dim,
            hidden_dims=(128, 128),
            output_dim=8,
        )
        self.continue_predictor = MultiLayerPerceptron(
            input_dim=self.model_state_dim,
            hidden_dims=(128, 128),
            output_dim=1,
        )

    def encode(self, observation, recurrent_state=None, no_grad=False):
        """
        Args:
            observation: (optional[batch, sequence], channel, height, width)
            recurrent_state: (optional[batch, sequence], recurrent_dim)
            no_grad: bool
        Returns:
            model_state:
        """
        pass

    def sequence(self, model_state, action):
        """
        Args:
            model_state: (optional[batch, sequence], )
            action: (optional[batch, sequence], )
        """
        pass

    def decode(self, model_state):
        """
        Args:
            model_state: (batch, sequence, ?)
        Returns:
            observation: (batch, sequence, channel, height, width)
        """
        pass

    def predict_reward(self, model_state):
        """
        Args:
            model_state: (batch, sequence, ?)
        Returns:
            observation: Normal(batch, sequence)
        """
        pass

    def predict_continue(self, model_state):
        """
        Args:
            model_state (batch, sequence, ?)
        Returns:
            observation: Bernoulli(batch, sequence)
        """
        logits = self.continue_predictor(model_state)
        logits = logits.squeeze(-1)
        distribution = torch.distributions.Bernoulli(logits=logits)
        return distribution
