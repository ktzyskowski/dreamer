from torch import nn

from .encoder import conv2d_output_size


# Decoder
# x_t ~ p(x_t | h_t, z_t)
class Decoder(nn.Module):
    def __init__(
        self,
        observation_shape: tuple,
        input_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        n_channels, h, w = observation_shape
        # mirror the encoder: compute the spatial size after 4 strided convolutions
        for _ in range(4):
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256 * h * w),
            nn.Unflatten(1, (256, h, w)),
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation(),
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation(),
            nn.ConvTranspose2d(
                32,
                n_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, latent):
        """Decode a latent vector into an image observation.

        Args:
            - latent: (batch, sequence, input_dim) tensor of latent vectors.

        Returns:
            - observation: (batch, sequence, channel, height, width) tensor of reconstructed observations.
        """
        # merge batch and sequence dimensions: (batch * sequence, input_dim)
        batch_size, sequence_length = latent.shape[0], latent.shape[1]
        latent = latent.view(batch_size * sequence_length, self.input_dim)
        # reconstruct observation: (batch * sequence, channel, height, width)
        observation = self.net(latent)
        # un-merge batch and sequence dimensions: (batch, sequence, channel, height, width)
        observation = observation.view(
            batch_size, sequence_length, *self.observation_shape
        )
        return observation
