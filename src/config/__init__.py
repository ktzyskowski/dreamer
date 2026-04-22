"""Typed configuration for the DreamerV3 training stack.

Each sub-module defines dataclasses scoped to one constructor's worth of args.
`Config` composes them. Dataclass defaults are the source of truth; the YAML
file at `conf/config.yaml` holds experiment overrides; `tyro.cli` layers CLI
overrides on top.
"""

from dataclasses import dataclass, field

from src.config.agent import ActorConfig, CriticConfig
from src.config.buffer import ReplayBufferConfig
from src.config.dreamer import DreamerConfig
from src.config.env import EnvironmentConfig
from src.config.losses import ActorCriticLossConfig, WorldModelLossConfig
from src.config.nets import CNNConfig, EncoderConfig, MLPConfig, resolve_activation
from src.config.torch import TorchConfig
from src.config.training import TrainingConfig
from src.config.twohot import TwoHotConfig
from src.config.utils import flatten, load_config
from src.config.world_model import RecurrentConfig, WorldModelConfig


@dataclass
class Config:
    torch: TorchConfig = field(default_factory=TorchConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    two_hot: TwoHotConfig = field(default_factory=TwoHotConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    reward_predictor: MLPConfig = field(default_factory=MLPConfig)
    continue_predictor: MLPConfig = field(default_factory=MLPConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    dreamer: DreamerConfig = field(default_factory=DreamerConfig)
    world_model_loss: WorldModelLossConfig = field(default_factory=WorldModelLossConfig)
    actor_critic_loss: ActorCriticLossConfig = field(
        default_factory=ActorCriticLossConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)


__all__ = [
    "Config",
    "ActorConfig",
    "ActorCriticLossConfig",
    "CNNConfig",
    "CriticConfig",
    "DreamerConfig",
    "EncoderConfig",
    "EnvironmentConfig",
    "MLPConfig",
    "RecurrentConfig",
    "ReplayBufferConfig",
    "TorchConfig",
    "TrainingConfig",
    "TwoHotConfig",
    "WorldModelConfig",
    "WorldModelLossConfig",
    "flatten",
    "load_config",
    "resolve_activation",
]
