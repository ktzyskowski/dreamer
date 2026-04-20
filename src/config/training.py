from dataclasses import dataclass


@dataclass
class TrainingConfig:
    n_steps: int = 10_000_000
    batch_size: int = 32
    sequence_length: int = 32
    warmup_steps: int = 1_024
    replay_ratio: int = 4
    grad_clip: float = 1000.0
    checkpoint_dir: str = "checkpoints"
    save_every_n_gradient_steps: int = 1_000
    checkpoint_path: str | None = None
