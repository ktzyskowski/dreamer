import logging
from typing import Type

import torch
import torch.nn as nn


def conv2d_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    if output_size <= 0:
        raise ValueError("Invalid CNN configuration.")
    return output_size


def conv_transpose2d_output_padding(in_size: int, target_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return target_size - ((in_size - 1) * stride - 2 * padding + kernel_size)


def compute_output_paddings(
    input_shape: tuple[int, int, int],
    kernel_size: int,
    stride: int,
    padding: int,
    channels: list[int],
) -> list[tuple[int, int]]:
    """Compute per-layer output_padding values for ConvTranspose2d layers.

    Args:
        input_shape: (n_channels, height, width) of the input image.
        kernel_size: Convolution kernel size (shared across all layers).
        stride: Convolution stride (shared across all layers).
        padding: Convolution padding (shared across all layers).
        channels: List of internal channel counts (not including input n_channels).

    Returns:
        Per-layer (op_h, op_w) for each ConvTranspose2d, ordered from outermost
        to innermost (same order as decoder layers).
    """
    _, height, width = input_shape

    heights = [height]
    widths = [width]
    h, w = height, width
    for _ in range(len(channels)):
        h = conv2d_output_size(h, kernel_size, stride, padding)
        w = conv2d_output_size(w, kernel_size, stride, padding)
        heights.append(h)
        widths.append(w)

    output_paddings = []
    for i in range(len(channels)):
        op_h = conv_transpose2d_output_padding(heights[-(i + 1)], heights[-(i + 2)], kernel_size, stride, padding)
        op_w = conv_transpose2d_output_padding(widths[-(i + 1)], widths[-(i + 2)], kernel_size, stride, padding)
        output_paddings.append((op_h, op_w))

    return output_paddings


def check_codec_compatibility(
    output_paddings: list[tuple[int, int]],
    stride: int,
) -> bool:
    """Check whether computed output_paddings are valid for nn.ConvTranspose2d.

    A configuration is compatible if every output_padding is in [0, stride),
    which is the constraint imposed by nn.ConvTranspose2d.

    Args:
        output_paddings: Per-layer (op_h, op_w) as returned by compute_output_paddings.
        stride: Convolution stride (shared across all layers).

    Returns:
        True if all output_padding values satisfy 0 <= op < stride.
    """
    return all(0 <= op_h < stride and 0 <= op_w < stride for op_h, op_w in output_paddings)


class ConvNet2D(nn.Module):
    """2D Convolutional Neural Net.

    Used as encoder inside DreamerV3 when trained on environments with image observations.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        channels: list[int],
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.n_channels = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.output_size = output_size

        if self.n_channels not in {1, 3}:
            logging.warning("n_channels is %d, double check input_shape given to CNN", self.n_channels)

        # prepend input channels so zip(channels, channels[1:]) produces all conv layer pairs
        channels = [self.n_channels] + list(channels)

        layers = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(activation())

        final_height, final_width = self.height, self.width
        for _ in range(len(channels) - 1):
            final_height = conv2d_output_size(final_height, kernel_size, stride, padding)
            final_width = conv2d_output_size(final_width, kernel_size, stride, padding)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels[-1] * final_height * final_width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN.

        Internally, all leading dimensions are flattened into a single batch dimension,
        then restored after processing through CNN.

        Args:
            x (*, channel, height, width): image tensors, with any leading dimensions.
        """
        x_flattened = x.flatten(end_dim=-4)
        encoding = self.net(x_flattened)
        encoding = encoding.unflatten(0, x.shape[:-3])
        return encoding


class ConvTransposeNet2D(nn.Module):
    """2D Transpose Convolutional Neural Net.

    Used as decoder inside DreamerV3 when trained on environments with image observations.
    The constructor arguments are same as the ConvNet2D to keep them compatible with each other.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        channels: list[int],
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.n_channels = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.output_size = output_size

        if self.n_channels not in {1, 3}:
            logging.warning("n_channels is %d, double check input_shape given to CNN", self.n_channels)

        output_paddings = compute_output_paddings(input_shape, kernel_size, stride, padding, channels)

        h, w = self.height, self.width
        for _ in range(len(channels)):
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)
        final_height, final_width = h, w

        # decoder channels: reversed internal channels, with n_channels as final output
        dec_channels = list(reversed(channels)) + [self.n_channels]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(output_size, dec_channels[0] * final_height * final_width))
        layers.append(nn.Unflatten(1, (dec_channels[0], final_height, final_width)))
        layers.append(activation())
        for i, (in_ch, out_ch) in enumerate(zip(dec_channels, dec_channels[1:])):
            layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding=output_paddings[i])
            )
            if i < len(dec_channels) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transpose CNN.

        Internally, all leading dimensions are flattened into a single batch dimension,
        then restored after processing through CNN.

        Args:
            x (*, output_size): encoded feature vectors, with any leading dimensions.
        """
        x_flattened = x.flatten(end_dim=-2)
        decoding = self.net(x_flattened)
        decoding = decoding.unflatten(0, x.shape[:-1])
        return decoding
