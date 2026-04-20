from dataclasses import dataclass, field

from src.config.nets import MLPConfig


@dataclass
class RecurrentConfig:
    n_blocks: int = 8


@dataclass
class WorldModelConfig:
    learning_rate: float = 0.0003
    recurrent_size: int = 128
    n_categoricals: int = 16
    n_classes: int = 16

    encoder: MLPConfig = field(default_factory=MLPConfig)
    decoder: MLPConfig = field(default_factory=MLPConfig)
    posterior_net: MLPConfig = field(default_factory=MLPConfig)
    prior_net: MLPConfig = field(default_factory=MLPConfig)
    reward_predictor: MLPConfig = field(default_factory=MLPConfig)
    continue_predictor: MLPConfig = field(default_factory=MLPConfig)
    recurrent_net: RecurrentConfig = field(default_factory=RecurrentConfig)
