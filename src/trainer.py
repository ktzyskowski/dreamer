import torch

from src.nn.functions import get_device


class Trainer:
    def __init__(self, world_model, actor, replay_buffer, config):
        self.device = get_device(priority=config.device)
        self.replay_buffer = replay_buffer
        self.world_model = world_model.to(self.device)
        self.actor = actor.to(self.device)

    def train(self):
        pass

    @torch.no_grad()
    def collect_experience(self, env, n_steps=256):
        """Collect experience from the given environment."""

        recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
        observation = env.reset()
        for _ in range(n_steps):
            full_state = self.world_model.get_full_state(observation.to(self.device), recurrent_state)
            action, _ = self.actor(full_state)
            action_idx = action.argmax(dim=-1).cpu().item()

            next_observation, reward, done = env.step(action_idx)
            self.replay_buffer.add(observation, action, reward, done)

            if done:
                recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
                observation = env.reset()
            else:
                recurrent_state = self.world_model.get_next_recurrent_state(full_state, action, recurrent_state)
                observation = next_observation
