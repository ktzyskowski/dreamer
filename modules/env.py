import gymnasium as gym
from gymnasium.wrappers import DtypeObservation, RescaleObservation, ResizeObservation
import numpy as np


class EnvironmentContext:
    def __init__(self):
        self.env = None

    def __enter__(self):
        self.env = gym.make("CarRacing-v3")
        self.env = ResizeObservation(self.env, (64, 64))
        self.env = DtypeObservation(self.env, dtype=np.float32)
        self.env = RescaleObservation(self.env, min_obs=np.float32(0.0), max_obs=np.float32(1.0))
        return self.env

    def __exit__(self, exc_type, exc, tb):
        if self.env:
            self.env.close()
            self.env = None
