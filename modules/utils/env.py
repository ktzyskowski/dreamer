import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    DtypeObservation,
    RescaleObservation,
    ResizeObservation,
    TransformObservation,
)
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch


class EnvironmentManager:
    def __init__(self, config):
        self.env_name = config.environment.name
        self.env_observation_size = tuple(config.environment.resize_observation)
        self.action_repeat = config.action_repeat
        self._env = None

    @property
    def env(self) -> gym.Env:
        if self._env is None:
            raise ValueError("Environment is not initialized")
        return self._env

    @property
    def action_size(self) -> int:
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            return int(action_space.n)
        if isinstance(action_space, gym.spaces.Box):
            return sum(action_space.shape)
        raise ValueError("Unsupported action space.")

    def __enter__(self):
        if self.env_name.startswith("ALE/"):
            import ale_py

            gym.register_envs(ale_py)

        self._env = gym.make(self.env_name)

        # downsize images
        self._env = ResizeObservation(self._env, self.env_observation_size)

        # change dtype from uint8 to float32 and downscale from 0,255 to 0,1
        self._env = DtypeObservation(self._env, dtype=np.float32)
        self._env = RescaleObservation(self._env, min_obs=np.float32(0.0), max_obs=np.float32(1.0))

        # CNN expects (channel, height, width), not (height, width, channel)
        old_observation_space = self._env.observation_space
        new_observation_space = gym.spaces.Box(
            low=np.moveaxis(old_observation_space.low, -1, 0),
            high=np.moveaxis(old_observation_space.high, -1, 0),
            dtype=old_observation_space.dtype,
        )
        self._env = TransformObservation(
            self._env,
            lambda observation: np.moveaxis(observation, -1, 0),
            new_observation_space,
        )

        # wrapping env in torch makes our lives easier, less manual conversions
        self._env = NumpyToTorch(self._env)

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
            observation, step_reward, step_done, step_truncated, _ = self.env.step(action)

            reward += float(step_reward)

            done = step_done or step_truncated
            if done:
                break

        return observation, reward, done

    def reset(self):
        observation, _ = self.env.reset()
        return observation
