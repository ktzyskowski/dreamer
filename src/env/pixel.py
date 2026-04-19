# placeholder file:
# old code for Atari envs, will refactor and introduce once moved on from CartPole and beginner test environments.


# import gymnasium as gym
# import numpy as np
# from gymnasium.wrappers import (
#     DtypeObservation,
#     RescaleObservation,
#     ResizeObservation,
#     TransformObservation,
# )
# from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
# from omegaconf import DictConfig


# class EnvironmentManager:
#     def __init__(self, config: DictConfig):
#         self.name = config.environment.name
#         self.env_type = config.environment.type  # "atari" or "vector"
#         self.action_repeat = config.environment.action_repeat
#         if self.env_type == "atari":
#             self.obs_shape = (config.environment.obs_height, config.environment.obs_width)
#         self._env = None

#     @property
#     def observation_space(self) -> gym.Space:
#         if self._env is None:
#             raise ValueError("Environment is not initialized")
#         return self._env.observation_space

#     @property
#     def action_space(self) -> gym.Space:
#         if self._env is None:
#             raise ValueError("Environment is not initialized")
#         return self._env.action_space

#     @property
#     def action_size(self) -> int:
#         if isinstance(self.action_space, gym.spaces.Discrete):
#             return int(self.action_space.n)
#         if isinstance(self.action_space, gym.spaces.Box):
#             return sum(self.action_space.shape)
#         raise ValueError("Unsupported action space.")

#     def __enter__(self):
#         if self.env_type == "atari":
#             import ale_py

#             gym.register_envs(ale_py)
#             self._env = gym.make(self.name)

#             # downsize images
#             self._env = ResizeObservation(self._env, self.obs_shape)

#             # change dtype from uint8 to float32 and downscale from 0,255 to 0,1
#             self._env = DtypeObservation(self._env, dtype=np.float32)
#             self._env = RescaleObservation(
#                 self._env, min_obs=np.float32(0.0), max_obs=np.float32(1.0)
#             )

#             # CNN expects (channel, height, width), not (height, width, channel)
#             old_observation_space = self._env.observation_space
#             if isinstance(old_observation_space, gym.spaces.Box):
#                 new_observation_space = gym.spaces.Box(
#                     low=np.moveaxis(old_observation_space.low, -1, 0),
#                     high=np.moveaxis(old_observation_space.high, -1, 0),
#                     dtype=old_observation_space.dtype.type,
#                 )
#                 self._env = TransformObservation(
#                     self._env,
#                     lambda observation: np.moveaxis(observation, -1, 0),
#                     new_observation_space,
#                 )
#         else:
#             self._env = gym.make(self.name)

#         # wrapping env in torch makes our lives easier, less manual conversions
#         self._env = NumpyToTorch(self._env)

#         return self

#     def __exit__(self, exc_type, exc, tb):
#         if self._env:
#             self._env.close()
#             self._env = None

#     def step(self, action):
#         if self._env is None:
#             raise ValueError("Environment is not initialized")

#         observation = None
#         reward = 0.0
#         done = False

#         # repeat given action specified number of times, accumulate
#         # rewards per step and discard intermediate observations.
#         for _ in range(self.action_repeat):
#             observation, step_reward, step_done, step_truncated, _ = self._env.step(
#                 action
#             )

#             reward += float(step_reward)

#             done = step_done or step_truncated
#             if done:
#                 break

#         return observation, reward, done

#     def reset(self):
#         if self._env is None:
#             raise ValueError("Environment is not initialized")

#         observation, _ = self._env.reset()
#         return observation
