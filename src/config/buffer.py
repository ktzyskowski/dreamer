from dataclasses import dataclass


@dataclass
class ReplayBufferConfig:
    capacity: int = 1_000_000
    dtype: str = "float32"
