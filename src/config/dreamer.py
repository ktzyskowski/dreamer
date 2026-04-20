from dataclasses import dataclass


@dataclass
class DreamerConfig:
    dream_horizon: int = 15
