from torch import nn

from .encoder import conv2d_output_size
from src.nets.mlp import MultiLayerPerceptron


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, output_size: int, hidden_dims: list):
        super().__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        self.net = MultiLayerPerceptron(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_size,
        )

    def forward(self, latent):
        return self.net(latent)


# TODO: output_padding is hardcoded to 1 on the last ConvTranspose2d layer to recover the pixel
# lost by floor division in the encoder (e.g. 64→31 via Conv2d can't be recovered by 31→64
# via ConvTranspose2d without output_padding=1). This fix only works for the current config
# (obs=64x64, kernel=3, stride=2, padding=0). The correct fix is to compute output_padding
# dynamically for each layer in __init__ based on the actual encoder sizes at each step:
#   natural_output(h_in) = (h_in - 1) * stride - 2 * padding + kernel_size
#   output_padding[i] = encoder_sizes[-(i+2)] - natural_output(encoder_sizes[-(i+1)])

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
        sizes = [h]
        # mirror the encoder: compute the spatial size after 4 strided convolutions
        for _ in range(4):
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)
            sizes.append(h)

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
                output_padding=1,  # fix padding issues
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
        # flatten any preceding dims: (..., input_dim)
        latent_shape = latent.shape
        latent = latent.view(-1, self.input_dim)
        # reconstruct observation: (..., channel, height, width)
        observation = self.net(latent)
        # un-flatten preceding dimensions: (..., channel, height, width)
        observation = observation.view(*latent_shape[:-1], *self.observation_shape)
        return observation
