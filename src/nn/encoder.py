import logging

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
        output_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.output_size = output_size
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
            nn.Linear(256 * h * w, output_size),
        )

    def forward(self, observation):
        input_shape = observation.shape
        # flatten all leading dims into a single batch dim for CNN processing
        observation = observation.view(-1, *self.observation_shape[-3:])
        encoding = self.cnn(observation)
        # restore original leading dims, replacing (C, H, W) with (output_size,)
        return encoding.view(*input_shape[:-3], self.output_size)
