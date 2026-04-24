import gymnasium as gym
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

from src.env.base import BaseEnv


class VectorEnv(BaseEnv):
    def _make_env(self) -> gym.Env:
        env = gym.make(self.name, **self.extra_kwargs)
        env = NumpyToTorch(env)
        return env
