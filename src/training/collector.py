from typing import Optional

import torch
import torch.nn as nn

from src.rl.dreamer import Dreamer
from src.data.buffer import ReplayBuffer
from src.env.base import BaseEnv
from src.util.probability import policy_distribution


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
        self.encoder = encoder
        self.world_model = world_model
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.device = device

        self._observation = self.env.reset()
        self._recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
        self._episode_return = 0.0
        self._episode_length = 0.0

    @torch.no_grad()
    def step(self) -> Optional[dict[str, int | float]]:
        encoded_observation = self.encoder(self._observation.to(self.device))
        latent_state = self.world_model.get_posterior_latent_state(encoded_observation, self._recurrent_state)
        full_state = get_full_state(latent_state, self._recurrent_state)
        action_logits = self.actor(full_state)
        policy = policy_distribution(action_logits, uniform_mix=0.01)
        action = policy.argmax(dim=-1).cpu().item()

        next_observation, reward, done = self.env.step(action)
        self.replay_buffer.add(self._observation, action, reward, done)
        self._episode_return += reward
        self._episode_length += 1

        if done:
            stats = {"env/episode_return": self._episode_return, "env/episode_length": self._episode_length}
            self._observation = self.env.reset()
            self._recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
            self._episode_return = 0.0
            self._episode_length = 0
            return stats

        self._recurrent_state = self.world_model.step(latent_state, self._recurrent_state, action)
        self._observation = next_observation
        return None
