from dataclasses import dataclass, field

from src.config.nets import EncoderConfig, MLPConfig


@dataclass
class RecurrentConfig:
    recurrent_size: int = 128
    n_blocks: int = 8


@dataclass
class WorldModelConfig:
    learning_rate: float = 0.0003
    n_categoricals: int = 16
    n_classes: int = 16

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    posterior_net: MLPConfig = field(default_factory=MLPConfig)
    prior_net: MLPConfig = field(default_factory=MLPConfig)
    recurrent_net: RecurrentConfig = field(default_factory=RecurrentConfig)
