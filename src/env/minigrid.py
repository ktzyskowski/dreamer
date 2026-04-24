import gymnasium as gym
import minigrid  # registers minigrid environments
import numpy as np
from gymnasium.core import Env
from gymnasium.wrappers import DtypeObservation, FlattenObservation
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

from src.env.base import BaseEnv


class MiniGridEnv(BaseEnv):
    def _make_env(self) -> Env:
        env = gym.make(self.name, **self.extra_kwargs)
        env = ImgObsWrapper(env)  # drop dict -> 7x7x3 int array
        env = DtypeObservation(env, np.float32)
        env = FlattenObservation(env)  # -> (147,)
        env = NumpyToTorch(env)
        return env
