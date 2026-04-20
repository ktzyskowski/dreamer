from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    type: str = "vector"
    name: str = "CartPole-v1"
    action_repeat: int = 1
