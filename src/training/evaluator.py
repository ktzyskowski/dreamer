from statistics import mean

import torch

from src.env.base import BaseEnv
from src.rl.dreamer import Dreamer


class Evaluator:
    """Run full episodes in a separate env to measure current policy performance.

    Uses the same `Dreamer.act` as the collector but does not write to the
    replay buffer and runs in a dedicated env so training exploration state
    is not disturbed.
    """

    def __init__(
        self,
        env: BaseEnv,
        dreamer: Dreamer,
        device: str,
        n_episodes: int = 5,
    ):
        self.env = env
        self.dreamer = dreamer
        self.device = device
        self.n_episodes = n_episodes

    @torch.no_grad()
    def run(self) -> dict[str, float]:
        recurrent_size = self.dreamer.world_model.recurrent_size
        returns: list[float] = []
        lengths: list[int] = []

        for _ in range(self.n_episodes):
            observation = self.env.reset()
            recurrent_state = torch.zeros(recurrent_size, device=self.device)
            episode_return = 0.0
            episode_length = 0
            done = False
            while not done:
                action, recurrent_state = self.dreamer.act(
                    observation.to(self.device), recurrent_state
                )
                observation, reward, done = self.env.step(action.argmax(dim=-1).item())
                episode_return += reward
                episode_length += 1
            returns.append(episode_return)
            lengths.append(episode_length)

        return {
            "eval/episode_return_mean": mean(returns),
            "eval/episode_length_mean": mean(lengths),
        }
