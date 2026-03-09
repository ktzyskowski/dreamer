from torch import nn

# full_state = cat[recurrent_state, latent_state]
# latent_posterior.shape == latent_prior.shape


# posterior = posterior_net(recurrent, encoded)
# prior = prior_net(recurrent)


class WorldModel(nn.Module):
    def __init__(
        self,
        observation_shape,
    ):
        super().__init__()

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

    def decode(self, latent_state):
        """
        Args:
            latent_state: (batch, sequence, ?)
        Returns:
            observation: (batch, sequence, channel, height, width)
        """
        pass

    def predict_reward(self, latent_state):
        """
        Args:
            latent_state: (batch, sequence, ?)
        Returns:
            observation: Normal(batch, sequence)
        """
        pass

    def predict_continue(self, latent_state):
        """
        Args:
            latent_state (batch, sequence, ?)
        Returns:
            observation: Bernoulli(batch, sequence)
        """
        pass
