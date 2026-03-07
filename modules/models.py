import torch
import torch.nn as nn


def count_parameters(model):
    """Count the number of trainable parameters in a given model.

    Args:
        - model (nn.Module)
    Returns:
        - n_parameters
    """
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters


class MultiLayerPerceptron(nn.Module):
    """Generic multi-layer perceptron class.

    Used throughout the Dreamer architecture for various components (reward predictor, continue predictor).
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.ReLU,
        output_activation=None,
    ):
        """Construct a new multi-layer perceptron.

        Args:
            input_dim (int): input layer size.
            hidden_dims (list[int]): list of hidden layer sizes.
            output_dim (int): output layer size.
            activation (nn.Module): hidden layer activation function. Defaults to nn.ReLU.
            output_activation (nn.Module): output layer activation function. Defaults to None.
        """
        super().__init__()
        layers = []

        # input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation())
        # hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(activation())
        # output layer + optional activation
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if output_activation:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        return y


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
            - model_state (batch, sequence, feature)
        Returns:
            - distribution (torch.Bernoulli)
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
            - model_state (batch, sequence, feature)
        Returns:
            - distribution (torch.Normal)
        """
        output = super().forward(x)
        mean, log_std = output.chunk(2, dim=-1)
        # (batch, sequence, 1) -> (batch, sequence)
        mean = mean.squeeze(-1)
        log_std = log_std.squeeze(-1)
        distribution = torch.distributions.Normal(mean, torch.exp(log_std))
        return distribution


# Sequence model
# h_t = f(h_t-1, z_t-1, a_t-1)
class SequenceModel(nn.Module):
    def __init__(self, hidden_dim, latent_dim, action_dim):
        super().__init__()


# Encoder
# e_t = f(x_t)
class Encoder(nn.Module):
    """Encoder model.

    Combines with posterior model to produce latent embedding z_t ~ p(z_t | h_t, x_t)
    """

    def __init__(
        self,
        observation_shape,
        output_dim,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.output_dim = output_dim
        self.stride = 2
        self.padding = 1
        self.kernel_size = 3

        n_channels = observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_channels,
                32,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                128,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                256,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                256
                * (observation_shape[1] // (self.stride**4))
                * (observation_shape[2] // (self.stride**4)),
                output_dim,
            ),
        )

    def forward(self, observation):
        """Encode an observation into a latent embedding.

        Args:
            - observation: (batch, sequence, channel, height, width) tensor of observations

        Returns:
            - embedding: (batch, sequence, embedding_dim) tensor of latent embeddings
        """
        # merge batch and sequence dimensions for CNN processing: (batch * sequence, channel, height, width)
        batch_size, sequence_length = observation.shape[0], observation.shape[1]
        observation = observation.view(
            batch_size * sequence_length, *self.observation_shape
        )
        # compute embedding for each observation: (batch * sequence, embedding_dim)
        embedding = self.cnn(observation)
        # un-merge batch and sequence dimensions: (batch, sequence, embedding_dim)
        embedding = embedding.view(batch_size, sequence_length, *self.observation_shape)
        return embedding


# Encoder p2
# z_t ~ p(z_t | h_t, e_t)
# e_t = f(x_t)
class PosteriorModel(nn.Module):
    def __init__(self):
        super().__init__()


# Dynamics predictor
# z_t ~ p(z_t | h_t)
class PriorModel(nn.Module):
    def __init__(self):
        super().__init__()


# Decoder
# x_t ~ p(x_t | h_t, z_t)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


class Actor(NormalModel):
    def __init__(self):
        super().__init__(input_dim=..., hidden_dims=..., activation=nn.ReLU)

    def forward(self, state):
        pass
