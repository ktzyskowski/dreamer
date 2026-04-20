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
