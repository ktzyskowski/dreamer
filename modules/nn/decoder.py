from torch import nn


# Decoder
# x_t ~ p(x_t | h_t, z_t)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
