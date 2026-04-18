from abc import ABC, abstractmethod

import gymnasium as gym


class BaseEnv(ABC):
    def __init__(self, name: str, action_repeat: int):
        self.name = name
        self.action_repeat = action_repeat
        self._env = None

    @property
    def observation_space(self) -> gym.Space:
        if self._env is None:
            raise ValueError("Environment is not initialized")
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        if self._env is None:
            raise ValueError("Environment is not initialized")
        return self._env.action_space

    @property
    def action_size(self) -> int:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return int(self.action_space.n)
        if isinstance(self.action_space, gym.spaces.Box):
            return sum(self.action_space.shape)
        raise ValueError("Unsupported action space.")

    @abstractmethod
    def _make_env(self) -> gym.Env: ...

    def __enter__(self):
        self._env = self._make_env()
        return self

    def __exit__(self, _):
        if self._env:
            self._env.close()
            self._env = None

    def step(self, action):
        if self._env is None:
            raise ValueError("Environment is not initialized")

        observation = None
        reward = 0.0
        done = False

        for _ in range(self.action_repeat):
            observation, step_reward, step_done, step_truncated, _ = self._env.step(
                action
            )
            reward += float(step_reward)
            done = step_done or step_truncated
            if done:
                break

        return observation, reward, done

    def reset(self):
        if self._env is None:
            raise ValueError("Environment is not initialized")
        observation, _ = self._env.reset()
        return observation
