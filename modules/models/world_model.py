from torch import nn
import torch

from modules.nn.encoder import Encoder
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
        # internal networks
        self.encoder = Encoder(
            observation_shape,
            output_dim=latent_dim,
            kernel_size=3,
            stride=2,
            padding=0,
        )
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

    def step(self, observation, action, recurrent_state):
        """
        Args:
            observation (*observation_shape):
            action (*action_shape):
            recurrent_state (recurrent_dim):
        Returns:
            recurrent_state (recurrent_dim):
            discrete_state (discrete_dim):
        """
        # add (batch, sequence) dimensions to single observation
        observation = observation.unsqueeze(0).unsqueeze(0)
        
        # encode observation
        latent = self.encoder(observation)
        
        # concatenate with recurrent state to get full model state
        model_state = torch.cat((latent, recurrent_state), dim=-1)

        return h, z

    def observe(self, observations, actions):
        pass

    def imagine(self):
        pass

    # def encode(self, observation, recurrent_state=None, no_grad=False):
    #     """
    #     Args:
    #         observation: (optional[batch, sequence], channel, height, width)
    #         recurrent_state: (optional[batch, sequence], recurrent_dim)
    #         no_grad: bool
    #     Returns:
    #         model_state:
    #     """
    #     pass

    # def sequence(self, model_state, action):
    #     """
    #     Args:
    #         model_state: (optional[batch, sequence], )
    #         action: (optional[batch, sequence], )
    #     """
    #     pass

    # def decode(self, model_state):
    #     """
    #     Args:
    #         model_state: (batch, sequence, ?)
    #     Returns:
    #         observation: (batch, sequence, channel, height, width)
    #     """
    #     pass

    # def predict_reward(self, model_state):
    #     """
    #     Args:
    #         model_state: (batch, sequence, ?)
    #     Returns:
    #         observation: Normal(batch, sequence)
    #     """
    #     pass

    # def predict_continue(self, model_state):
    #     """
    #     Args:
    #         model_state (batch, sequence, ?)
    #     Returns:
    #         observation: Bernoulli(batch, sequence)
    #     """
    #     logits = self.continue_predictor(model_state)
    #     logits = logits.squeeze(-1)
    #     distribution = torch.distributions.Bernoulli(logits=logits)
    #     return distribution
