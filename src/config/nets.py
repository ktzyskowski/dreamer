from dataclasses import dataclass, field
from typing import Type

import torch.nn as nn

from src.nets.activations import RMSNormSiLU

ACTIVATIONS: dict[str, Type[nn.Module]] = {
    "rmsnorm_silu": RMSNormSiLU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def resolve_activation(name: str) -> Type[nn.Module]:
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Known: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]


@dataclass
class MLPConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "rmsnorm_silu"


@dataclass
class CNNConfig:
    padding: int = 1
    stride: int = 1
    kernel_size: int = 3
    activation: str = "rmsnorm_silu"
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class EncoderConfig:
    mlp: MLPConfig = field(default_factory=MLPConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
