from typing import Optional

import torch

from src.data.buffer import ReplayBuffer
from src.env.base import BaseEnv
from src.rl.dreamer import Dreamer


class Collector:
    """Collector class drives the environment interaction loop."""

    def __init__(
        self,
        env: BaseEnv,
        dreamer: Dreamer,
        replay_buffer: ReplayBuffer,
        device: str,
    ):
        self.env = env
        self.dreamer = dreamer
        self.replay_buffer = replay_buffer
        self.device = device

        self._observation = self.env.reset()
        self._recurrent_state = torch.zeros(self.dreamer.world_model.recurrent_size, device=self.device)
        self._episode_return = 0.0
        self._episode_length = 0

    @torch.no_grad()
    def step(self) -> Optional[dict[str, int | float]]:
        action, next_recurrent_state = self.dreamer.act(
            self._observation.to(self.device),
            self._recurrent_state,
        )

        next_observation, reward, done = self.env.step(action.argmax(dim=-1).item())
        self.replay_buffer.add(self._observation, action, reward, done)
        self._episode_return += reward
        self._episode_length += 1

        if done:
            stats = {"env/episode_return": self._episode_return, "env/episode_length": self._episode_length}
            self._observation = self.env.reset()
            self._recurrent_state = torch.zeros(self.dreamer.world_model.recurrent_size, device=self.device)
            self._episode_return = 0.0
            self._episode_length = 0
            return stats

        self._recurrent_state = next_recurrent_state
        self._observation = next_observation
        return None
