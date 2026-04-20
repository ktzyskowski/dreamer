from dataclasses import dataclass


@dataclass
class TwoHotConfig:
    low: float = -20.0
    high: float = 20.0
    n_bins: int = 255
