class Collector:
    def __init__(self):
        pass


# import torch

# from src.agent.actor import DiscreteActor
# from src.data.buffer import ReplayBuffer
# from src.util.env import EnvironmentManager
# from world_model import WorldModel


# class Collector:
#     """Drives the environment loop and writes transitions to the replay buffer.

#     Owns all per-rollout state: current observation, recurrent state, and
#     episode accumulators. A single `step()` call takes exactly one env step
#     (after action repeat), adds the transition to the buffer, and returns
#     episode stats on done-boundaries — else `None`.
#     """

#     def __init__(
#         self,
#         env: EnvironmentManager,
#         world_model: WorldModel,
#         actor: DiscreteActor,
#         replay_buffer: ReplayBuffer,
#         device: str,
#     ):
#         self.env = env
#         self.world_model = world_model
#         self.actor = actor
#         self.replay_buffer = replay_buffer
#         self.device = device

#         self._observation = self.env.reset()
#         self._recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
#         self._episode_reward = 0.0
#         self._episode_length = 0

#     @torch.no_grad()
#     def step(self) -> dict | None:
#         encoded_observation = self.world_model.encoder(self._observation.to(self.device))
#         full_state, _ = self.world_model.get_full_state(encoded_observation, self._recurrent_state)
#         action, _ = self.actor(full_state)
#         action_idx = action.argmax(dim=-1).cpu().item()

#         next_observation, reward, done = self.env.step(action_idx)
#         self.replay_buffer.add(self._observation, action, reward, done)

#         self._episode_reward += reward
#         self._episode_length += 1

#         if done:
#             stats = {
#                 "env/episode_reward": self._episode_reward,
#                 "env/episode_length": self._episode_length,
#             }
#             self._observation = self.env.reset()
#             self._recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
#             self._episode_reward = 0.0
#             self._episode_length = 0
#             return stats

#         self._recurrent_state = self.world_model.get_next_recurrent_state(
#             full_state, action, self._recurrent_state
#         )
#         self._observation = next_observation
#         return None
