import gymnasium as gym
import minigrid  # registers minigrid environments
import numpy as np
from gymnasium.core import Env, ObservationWrapper
from gymnasium.wrappers import DtypeObservation
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from minigrid.wrappers import ImgObsWrapper

from src.env.base import BaseEnv


class ChannelFirst(ObservationWrapper):
    """Convert (H, W, C) observations to (C, H, W) for PyTorch CNNs."""

    def __init__(self, env: Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.transpose(2, 0, 1),
            high=env.observation_space.high.transpose(2, 0, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.transpose(2, 0, 1)


class MiniGridEnv(BaseEnv):
    def _make_env(self) -> Env:
        env = gym.make(self.name, **self.extra_kwargs)
        env = ImgObsWrapper(env)  # drop dict -> (7, 7, 3)
        env = DtypeObservation(env, np.float32)
        env = ChannelFirst(env)  # (7, 7, 3) -> (3, 7, 7)
        env = NumpyToTorch(env)
        return env
