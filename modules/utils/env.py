from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    DtypeObservation,
    RescaleObservation,
    ResizeObservation,
    TransformObservation,
)


class EnvironmentManager:
    def __init__(self, action_repeat=1):
        self.action_repeat = action_repeat
        self._env = None
        self._observation_space: Optional[gym.Space] = None
        self._action_space: Optional[gym.Space] = None

    @property
    def env(self):
        if self._env is None:
            raise ValueError("Environment is not initialized")
        return self._env

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise ValueError("Environment is not initialized")
        return self._observation_space

    @property
    def action_space(self):
        if self._action_space is None:
            raise ValueError("Environment is not initialized")
        return self._action_space

    def __enter__(self):
        self._env = gym.make("CarRacing-v3")
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

        # downsize images to (64, 64)
        self._env = ResizeObservation(self._env, (64, 64))
        # change dtype from uint8 to float32 and downscale from 0,255 to 0,1
        self._env = DtypeObservation(self._env, dtype=np.float32)
        self._env = RescaleObservation(
            self._env, min_obs=np.float32(0.0), max_obs=np.float32(1.0)
        )
        # rearrange observation axes from (height, width, channel) to (channel, height, width)
        # as expected by CNN layers
        self._env = TransformObservation(
            self._env,
            lambda observation: np.moveaxis(observation, -1, 0),
            self._env.observation_space,
        )

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._env:
            self._env.close()
            self._env = None
            self._observation_space = None
            self._action_space = None

    def step(self, action):
        observation = None
        reward = 0.0
        done = False

        # repeat given action specified number of times, accumulate
        # rewards per step and discard intermediate observations.
        for _ in range(self.action_repeat):
            observation, step_reward, step_done, step_truncated, _ = self.env.step(
                action
            )

            reward += float(step_reward)

            done = step_done or step_truncated
            if done:
                break

        return observation, reward, done

    def reset(self):
        observation, _ = self.env.reset()
        return observation
