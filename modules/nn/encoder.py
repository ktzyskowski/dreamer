from torch import nn


def conv2d_output_size(input_size: int, kernel_size: int, stride: int, padding: int):
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    if output_size <= 0:
        raise ValueError("Invalid CNN configuration.")
    return output_size


# Encoder
# Combines with posterior model to produce latent embedding z_t ~ p(z_t | h_t, f(x_t))
# e_t = f(x_t)
class Encoder(nn.Module):
    def __init__(
        self,
        observation_shape: tuple,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        n_channels, h, w = observation_shape
        # we need to calculate final (h, w) of images after all conv layers to calculate linear features
        for _ in range(4):
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_channels,
                32,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            activation(),
            nn.Conv2d(
                32,
                64,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            activation(),
            nn.Conv2d(
                64,
                128,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            activation(),
            nn.Conv2d(
                128,
                256,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            activation(),
            nn.Flatten(),
            nn.Linear(256 * h * w, output_dim),
        )

    def forward(self, observation):
        """Encode an image observation.

        Args:
            - observation: (batch, sequence, channel, height, width) tensor of observations.

        Returns:
            - embedding: (batch, sequence, output_dim) tensor of latent embeddings.
        """
        # merge batch and sequence dimensions for CNN processing: (batch * sequence, channel, height, width)
        input_shape = observation.shape
        if input_shape > 4:
            batch_size, sequence_length = input_shape[0], input_shape[1]
            observation = observation.view(
                batch_size * sequence_length, *self.observation_shape
            )
        # compute embedding for each observation: (batch * sequence, output_dim)
        latent = self.cnn(observation)
        # un-merge batch and sequence dimensions: (batch, sequence, output_dim)
        if input_shape > 4:
            latent = latent.view(batch_size, sequence_length, self.output_dim)
        return latent
