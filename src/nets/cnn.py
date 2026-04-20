# placeholder for when I move to pixel environments and need a CNN encoder/decoder

# from torch import nn


# def conv2d_output_size(input_size: int, kernel_size: int, stride: int, padding: int):
#     output_size = (input_size + 2 * padding - kernel_size) // stride + 1
#     if output_size <= 0:
#         raise ValueError("Invalid CNN configuration.")
#     return output_size


# # Encoder
# # Combines with posterior model to produce latent embedding z_t ~ p(z_t | h_t, f(x_t))
# # e_t = f(x_t)
# class ConvNet2D(nn.Module):
#     def __init__(
#         self,
#         observation_shape: tuple,
#         output_size: int,
#         kernel_size: int,
#         stride: int,
#         padding: int,
#         activation=nn.ReLU,
#     ):
#         super().__init__()
#         self.observation_shape = observation_shape
#         self.output_size = output_size
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#         n_channels, h, w = observation_shape
#         # we need to calculate final (h, w) of images after all conv layers to calculate linear features
#         for _ in range(4):
#             h = conv2d_output_size(h, kernel_size, stride, padding)
#             w = conv2d_output_size(w, kernel_size, stride, padding)

#         self.cnn = nn.Sequential(
#             nn.Conv2d(
#                 n_channels,
#                 32,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.padding,
#             ),
#             activation(),
#             nn.Conv2d(
#                 32,
#                 64,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.padding,
#             ),
#             activation(),
#             nn.Conv2d(
#                 64,
#                 128,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.padding,
#             ),
#             activation(),
#             nn.Conv2d(
#                 128,
#                 256,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.padding,
#             ),
#             activation(),
#             nn.Flatten(),
#             nn.Linear(256 * h * w, output_size),
#         )

#     def forward(self, observation):
#         input_shape = observation.shape
#         # flatten all leading dims into a single batch dim for CNN processing
#         observation = observation.view(-1, *self.observation_shape[-3:])
#         encoding = self.cnn(observation)
#         # restore original leading dims, replacing (C, H, W) with (output_size,)
#         return encoding.view(*input_shape[:-3], self.output_size)

# old decoder.py

# from torch import nn

# from .cnn import conv2d_output_size

# # TODO: output_padding is hardcoded to 1 on the last ConvTranspose2d layer to recover the pixel
# # lost by floor division in the encoder (e.g. 64→31 via Conv2d can't be recovered by 31→64
# # via ConvTranspose2d without output_padding=1). This fix only works for the current config
# # (obs=64x64, kernel=3, stride=2, padding=0). The correct fix is to compute output_padding
# # dynamically for each layer in __init__ based on the actual encoder sizes at each step:
# #   natural_output(h_in) = (h_in - 1) * stride - 2 * padding + kernel_size
# #   output_padding[i] = encoder_sizes[-(i+2)] - natural_output(encoder_sizes[-(i+1)])


# # Decoder
# # x_t ~ p(x_t | h_t, z_t)
# class Decoder(nn.Module):
#     def __init__(
#         self,
#         observation_shape: tuple,
#         input_dim: int,
#         kernel_size: int,
#         stride: int,
#         padding: int,
#         activation=nn.ReLU,
#     ):
#         super().__init__()
#         self.observation_shape = observation_shape
#         self.input_dim = input_dim
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#         n_channels, h, w = observation_shape
#         sizes = [h]
#         # mirror the encoder: compute the spatial size after 4 strided convolutions
#         for _ in range(4):
#             h = conv2d_output_size(h, kernel_size, stride, padding)
#             w = conv2d_output_size(w, kernel_size, stride, padding)
#             sizes.append(h)

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256 * h * w),
#             nn.Unflatten(1, (256, h, w)),
#             nn.ConvTranspose2d(
#                 256,
#                 128,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#             ),
#             activation(),
#             nn.ConvTranspose2d(
#                 128,
#                 64,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#             ),
#             activation(),
#             nn.ConvTranspose2d(
#                 64,
#                 32,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#             ),
#             activation(),
#             nn.ConvTranspose2d(
#                 32,
#                 n_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 output_padding=1,  # fix padding issues
#             ),
#             nn.Sigmoid(),
#         )

#     def forward(self, latent):
#         """Decode a latent vector into an image observation.

#         Args:
#             - latent: (batch, sequence, input_dim) tensor of latent vectors.

#         Returns:
#             - observation: (batch, sequence, channel, height, width) tensor of reconstructed observations.
#         """
#         # flatten any preceding dims: (..., input_dim)
#         latent_shape = latent.shape
#         latent = latent.view(-1, self.input_dim)
#         # reconstruct observation: (..., channel, height, width)
#         observation = self.net(latent)
#         # un-flatten preceding dimensions: (..., channel, height, width)
#         observation = observation.view(*latent_shape[:-1], *self.observation_shape)
#         return observation
