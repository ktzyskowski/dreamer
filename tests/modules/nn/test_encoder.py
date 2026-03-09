import torch
from torch import nn

from modules.nn.encoder import Encoder


def test_encoder_shapes():
    batch_size = 12
    sequence_length = 16
    encoder_kwargs = {
        "observation_shape": (64, 64, 3),
        "output_dim": 128,
        "stride": 1,
        "padding": 1,
        "kernel_size": 2,
        "activation": nn.ReLU,
    }

    encoder = Encoder(**encoder_kwargs)
    observation = torch.rand(
        (batch_size, sequence_length, *encoder_kwargs["observation_shape"])
    )
    encoding = encoder(observation)
    assert encoding.shape == (batch_size, sequence_length, encoder_kwargs["output_dim"])
