from dataclasses import dataclass, field

from src.config.nets import MLPConfig


@dataclass
class ActorConfig:
    learning_rate: float = 0.0003
    net: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class CriticConfig:
    learning_rate: float = 0.0001
    ema_decay: float = 0.98
    net: MLPConfig = field(default_factory=MLPConfig)
