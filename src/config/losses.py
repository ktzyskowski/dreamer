from dataclasses import dataclass


@dataclass
class WorldModelLossConfig:
    beta_posterior: float = 0.1
    beta_prior: float = 1.0
    beta_prediction: float = 1.0
    free_nats: float = 1.0


@dataclass
class ActorCriticLossConfig:
    discount: float = 0.99
    trace_decay: float = 0.95
    entropy_coefficient: float = 0.05
    slow_regularization_weight: float = 1.0
    advantage_norm_decay: float = 0.99
