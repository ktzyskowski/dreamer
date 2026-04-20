from dataclasses import dataclass


@dataclass
class TorchConfig:
    device: str = "cuda"
    float32_matmul_precision: str | None = "high"
