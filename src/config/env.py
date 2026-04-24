from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentConfig:
    type: str = "vector"
    name: str = "CartPole-v1"
    action_repeat: int = 1
    # arbitrary extra kwargs forwarded to gym.make (e.g. max_episode_steps, render_mode)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
