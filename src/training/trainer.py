class Trainer:
    def __init__(self):
        pass


# import logging

# import torch

# from src.agent.actor import DiscreteActor
# from src.agent.critic import DualCritic
# from src.data.buffer import ReplayBuffer
# from src.losses.actor_critic import ActorCriticLoss
# from src.losses.world_model import WorldModelLoss
# from src.training.checkpoint import CheckpointManager
# from src.training.collector import Collector
# from src.training.metrics import MetricsAggregator
# from src.util.device import get_device
# from src.util.env import EnvironmentManager
# from world_model import WorldModel


# class Trainer:
#     def __init__(
#         self,
#         env: EnvironmentManager,
#         world_model: WorldModel,
#         actor: DiscreteActor,
#         critic: DualCritic,
#         replay_buffer: ReplayBuffer,
#         metrics: MetricsAggregator,
#         config,
#     ):
#         self.replay_buffer = replay_buffer
#         self.gradient_step_counter = 0

#         self.device = get_device(priority=config.device)
#         logging.info("Using device %s", self.device)

#         self.warmup_steps = config.warmup_steps
#         self.batch_size = config.batch_size
#         self.sequence_length = config.sequence_length

#         # paper definition: replay_ratio = timesteps_trained / env_timesteps (pre-action-repeat)
#         # => env steps per gradient step = batch_timesteps * action_repeat / replay_ratio
#         action_repeat = config.environment.action_repeat
#         self.env_steps_per_gradient_step = max(
#             1, (self.batch_size * self.sequence_length * action_repeat) // config.replay_ratio
#         )
#         logging.info("Env steps per gradient step: %d", self.env_steps_per_gradient_step)

#         self.world_model = torch.compile(world_model.to(self.device))
#         self.actor = torch.compile(actor.to(self.device))
#         self.critic = torch.compile(critic.to(self.device))

#         self.world_model_optimizer = torch.optim.Adam(
#             world_model.parameters(), lr=config.world_model.learning_rate
#         )
#         self.actor_optimizer = torch.optim.Adam(
#             actor.parameters(), lr=config.actor.learning_rate
#         )
#         self.critic_optimizer = torch.optim.Adam(
#             critic.fast.parameters(), lr=config.critic.learning_rate
#         )

#         self.world_model_loss = WorldModelLoss(config).to(self.device)
#         self.actor_critic_loss = ActorCriticLoss(config).to(self.device)

#         self.metrics = metrics
#         self.collector = Collector(
#             env=env,
#             world_model=self.world_model,
#             actor=self.actor,
#             replay_buffer=replay_buffer,
#             device=self.device,
#         )
#         self.checkpointer = CheckpointManager(
#             directory="checkpoints",
#             modules={
#                 "world_model": self.world_model,
#                 "actor": self.actor,
#                 "critic": self.critic,
#                 "world_model_optimizer": self.world_model_optimizer,
#                 "actor_optimizer": self.actor_optimizer,
#                 "critic_optimizer": self.critic_optimizer,
#             },
#             save_every_n_gradient_steps=1_000,
#         )

#     def _batch_to_device(self, batch):
#         return {
#             k: torch.from_numpy(v).to(dtype=torch.float32, device=self.device, non_blocking=True)
#             for k, v in batch.items()
#         }

#     def train(self, n_steps=256, start_step=0, start_gradient_step=0):
#         self.gradient_step_counter = start_gradient_step

#         for step in range(start_step, start_step + n_steps):
#             episode_stats = self.collector.step()
#             if episode_stats is not None:
#                 self.metrics.log(episode_stats, step=step)

#             if step >= self.warmup_steps and step % self.env_steps_per_gradient_step == 0:
#                 self.gradient_step()

#             self.checkpointer.maybe_save(step=step, gradient_step=self.gradient_step_counter)

#     def gradient_step(self):
#         batch = self.replay_buffer.sample(
#             batch_size=self.batch_size, sequence_length=self.sequence_length
#         )
#         batch = self._batch_to_device(batch)

#         self.world_model_optimizer.zero_grad()
#         observed_output = self.world_model.observe(batch)
#         world_model_loss = self.world_model_loss(batch, observed_output)
#         world_model_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1000)
#         self.world_model_optimizer.step()

#         recurrent_states = observed_output["recurrent_states"].detach()
#         self.actor_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()

#         self.world_model.requires_grad_(False)
#         dream_output = self.world_model.dream(batch, recurrent_states, self.actor)
#         self.world_model.requires_grad_(True)

#         actor_critic_loss = self.actor_critic_loss(dream_output, self.critic)
#         actor_critic_loss.backward()

#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1000)
#         torch.nn.utils.clip_grad_norm_(self.critic.fast.parameters(), 1000)
#         self.actor_optimizer.step()
#         self.critic_optimizer.step()
#         self.critic.update_slow()

#         self.metrics.update({**self.world_model_loss.metrics, **self.actor_critic_loss.metrics})
#         self.gradient_step_counter += 1
#         self.metrics.maybe_flush(self.gradient_step_counter)

#         if self.gradient_step_counter % 25 == 0:
#             logging.info("Gradient steps performed: %d", self.gradient_step_counter)
