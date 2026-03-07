import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Generic multi-layer perceptron class.

    Used throughout the Dreamer architecture for various components (e.g. reward predictor, continue predictor, etc.).
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers = []
        # input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation())
        # hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(activation())
        # output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # optional output activation
        if output_activation:
            layers.append(output_activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        return y


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
        latent_dim,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.latent_dim = latent_dim
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
                256 * (observation_shape[1] // (self.stride**4)) * (observation_shape[2] // (self.stride**4)),
                latent_dim,
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
        observation = observation.view(batch_size * sequence_length, *self.observation_shape)
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


# Reward predictor
# r_t ~ p(r_t | h_t, z_t)
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()


# Continue predictor
# c_t ~ p(c_t | h_t, z_t)
class ContinueModel(nn.Module):
    def __init__(self):
        super().__init__()


# Decoder
# x_t ~ p(x_t | h_t, z_t)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
