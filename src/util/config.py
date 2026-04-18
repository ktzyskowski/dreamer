"""Typed configuration for the DreamerV3 training stack.

Each section maps 1:1 to a block in `conf/config.yaml`. Dataclass defaults
are the source of truth; the YAML file holds experiment-specific overrides;
`tyro.cli(Config, default=...)` layers CLI overrides on top of both.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro
import yaml


@dataclass
class TorchConfig:
    float32_matmul_precision: str | None = "high"


@dataclass
class EnvironmentConfig:
    name: str = "CartPole-v1"
    type: str = "vector"
    action_repeat: int = 1
    obs_width: int = 64
    obs_height: int = 64


@dataclass
class TwoHotConfig:
    n_bins: int = 255
    low: float = -20.0
    high: float = 20.0


@dataclass
class ReplayBufferConfig:
    capacity: int = 1_000_000
    dtype: str = "float32"


@dataclass
class EncoderConfig:
    kernel_size: int = 3
    stride: int = 2
    padding: int = 0
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class MLPConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class RecurrentModelConfig:
    n_blocks: int = 8


@dataclass
class WorldModelConfig:
    learning_rate: float = 3e-4
    dream_horizon: int = 15
    n_dreams: int = 64
    recurrent_size: int = 128
    n_categoricals: int = 16
    n_classes: int = 16
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    posterior_net: MLPConfig = field(default_factory=MLPConfig)
    prior_net: MLPConfig = field(default_factory=MLPConfig)
    reward_predictor: MLPConfig = field(default_factory=MLPConfig)
    continue_predictor: MLPConfig = field(default_factory=MLPConfig)
    recurrent_model: RecurrentModelConfig = field(default_factory=RecurrentModelConfig)


@dataclass
class ActorConfig:
    learning_rate: float = 3e-4
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class CriticConfig:
    learning_rate: float = 1e-4
    ema_decay: float = 0.98
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class WorldModelLossConfig:
    beta_posterior: float = 0.1
    beta_prior: float = 1.0
    beta_prediction: float = 1.0
    free_nats: float = 1.0


@dataclass
class ActorCriticLossConfig:
    ema_decay: float = 0.99
    eta: float = 5e-2
    lamda: float = 0.95
    gamma: float = 0.99
    slow_reg: float = 1.0


@dataclass
class Config:
    device: str = "cuda"
    warmup_steps: int = 1_024
    replay_ratio: int = 4
    batch_size: int = 32
    sequence_length: int = 32
    checkpoint_path: str | None = None
    torch: TorchConfig = field(default_factory=TorchConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    two_hot: TwoHotConfig = field(default_factory=TwoHotConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    world_model_loss: WorldModelLossConfig = field(default_factory=WorldModelLossConfig)
    actor_critic_loss: ActorCriticLossConfig = field(
        default_factory=ActorCriticLossConfig
    )


def _build(cls: type, data: Any) -> Any:
    """Recursively construct a dataclass instance from a nested dict."""
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _build(f.type, data[f.name])
    return cls(**kwargs)


def load_config(yaml_path: str | Path = "conf/config.yaml") -> Config:
    """Load defaults from YAML, then apply CLI overrides via tyro."""
    path = Path(yaml_path)
    default = Config()
    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        default = _build(Config, raw)
    return tyro.cli(Config, default=default)


def flatten(d: dict, sep="."):
    """Flatten the given dictionary."""

    def flatten_helper(d: dict, prefix: str = ""):
        items = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(flatten_helper(v, key))
            else:
                items[key] = v
        return items

    return flatten_helper(d)
