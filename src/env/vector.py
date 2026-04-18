import gymnasium as gym
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

from src.env.base import BaseEnv


class VectorEnv(BaseEnv):
    def __init__(self, name: str, action_repeat: int = 1):
        super().__init__(name, action_repeat)

    def _make_env(self) -> gym.Env:
        env = gym.make(self.name)
        env = NumpyToTorch(env)
        return env
