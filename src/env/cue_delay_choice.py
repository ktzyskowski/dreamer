import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

from src.env.base import BaseEnv


class CueDelayChoiceEnv(gym.Env):
    """
    Memory test: cue -> delay -> choice
    """

    def __init__(self, n_actions=4, delay_steps=5):
        super().__init__()

        self.n_actions = n_actions
        self.delay_steps = delay_steps

        # observation: cue one-hot OR blank OR go-signal
        self.obs_dim = n_actions

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.action_space = spaces.Discrete(n_actions)

        self.phase = None
        self.cue = None
        self.t = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cue = self.np_random.integers(self.n_actions)

        self.phase = "cue"
        self.t = 0

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if self.phase == "cue":
            self.phase = "delay"
            self.t = 0

        elif self.phase == "delay":
            self.t += 1
            if self.t >= self.delay_steps:
                self.phase = "choice"

        elif self.phase == "choice":
            reward = float(action == self.cue)
            terminated = True
            self.phase = "done"

        obs = self._get_obs()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        if self.phase == "cue":
            obs = np.zeros(self.n_actions, dtype=np.float32)
            obs[self.cue] = 1.0

        elif self.phase == "delay":
            obs = np.zeros(self.n_actions, dtype=np.float32)

        elif self.phase == "choice":
            obs = np.ones(self.n_actions, dtype=np.float32) * 0.5

        else:
            obs = np.zeros(self.n_actions, dtype=np.float32)

        return obs


class CueDelayChoiceVectorEnv(BaseEnv):
    def _make_env(self) -> gym.Env:
        env = CueDelayChoiceEnv(**self.extra_kwargs)
        env = NumpyToTorch(env)
        return env
