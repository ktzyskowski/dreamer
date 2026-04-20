import logging

import torch
import torch.nn as nn

from src.data.buffer import ReplayBuffer
from src.losses.actor_critic import ActorCriticLoss
from src.losses.world_model import WorldModelLoss
from src.rl.dreamer import Dreamer
from src.training.checkpoint import CheckpointManager
from src.training.collector import Collector
from src.training.metrics import MetricsAggregator


class Trainer:
    def __init__(
        self,
        dreamer: Dreamer,
        collector: Collector,
        replay_buffer: ReplayBuffer,
        metrics: MetricsAggregator,
        world_model_loss: WorldModelLoss,
        actor_critic_loss: ActorCriticLoss,
        device: str,
        batch_size: int = 32,
        sequence_length: int = 32,
        warmup_steps: int = 1_024,
        replay_ratio: int = 4,
        action_repeat: int = 1,
        world_model_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-4,
        grad_clip: float = 1000.0,
        checkpoint_dir: str = "checkpoints",
        save_every_n_gradient_steps: int = 1_000,
    ):
        self.device = device
        self.dreamer = dreamer
        self.collector = collector
        self.replay_buffer = replay_buffer
        self.metrics = metrics

        self.world_model_loss = world_model_loss.to(device)
        self.actor_critic_loss = actor_critic_loss.to(device)

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip

        # replay_ratio = timesteps_trained / env_timesteps (pre-action-repeat)
        self.env_steps_per_gradient_step = max(
            1, (batch_size * sequence_length * action_repeat) // replay_ratio
        )
        logging.info(
            "Env steps per gradient step: %d", self.env_steps_per_gradient_step
        )

        # Optimizers -------------------------------------------------------- #

        self.world_model_optimizer = torch.optim.Adam(
            dreamer.world_model_parameters(), lr=world_model_lr
        )
        self.actor_optimizer = torch.optim.Adam(dreamer.actor_parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            dreamer.critic_parameters(), lr=critic_lr
        )

        # Checkpointing ----------------------------------------------------- #

        self.checkpointer = CheckpointManager(
            directory=checkpoint_dir,
            modules={
                "dreamer": dreamer,
                "world_model_optimizer": self.world_model_optimizer,
                "actor_optimizer": self.actor_optimizer,
                "critic_optimizer": self.critic_optimizer,
            },
            save_every_n_gradient_steps=save_every_n_gradient_steps,
        )

    def train(self, n_steps: int, start_step: int = 0, start_gradient_step: int = 0):
        gradient_step = start_gradient_step
        for step in range(start_step, start_step + n_steps):
            episode_stats = self.collector.step()
            if episode_stats is not None:
                # episode is terminated, log episode return and length
                self.metrics.log(episode_stats, step=step)

            ready = (
                step >= self.warmup_steps
                and step % self.env_steps_per_gradient_step == 0
            )
            if ready:
                self.gradient_step(gradient_step)
                self.checkpointer.maybe_save(step=step, gradient_step=gradient_step)
                gradient_step += 1

    def gradient_step(self, step: int):
        batch = self.replay_buffer.sample_torch(
            self.batch_size, self.sequence_length, self.device
        )

        # World Model -------------------------------------------------- #
        self.world_model_optimizer.zero_grad()
        observed_output = self.dreamer.observe(batch)
        world_model_loss, world_model_metrics = self.world_model_loss(
            batch, observed_output
        )
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.dreamer.world_model_parameters(), self.grad_clip)
        self.world_model_optimizer.step()

        # Actor / Critic ----------------------------------------------- #
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # freeze world-model side so actor-critic gradients don't leak through
        # reward_predictor / continue_predictor / world_model into its params
        self.dreamer.freeze_world_model()
        dream_output = self.dreamer.dream(
            full_states=observed_output["full_states"].detach(),
            recurrent_states=observed_output["recurrent_states"].detach(),
        )
        self.dreamer.unfreeze_world_model()

        critic = self.dreamer.agent.critic
        full_states = dream_output["full_states"]
        fast_critic_logits = critic.fast(full_states)
        slow_critic_logits = critic.slow(full_states)
        actor_critic_loss, actor_critic_metrics = self.actor_critic_loss(
            dream_output, fast_critic_logits, slow_critic_logits
        )
        actor_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.dreamer.actor_parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.dreamer.critic_parameters(), self.grad_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.dreamer.agent.critic.update_slow()

        # Bookkeeping -------------------------------------------------- #
        self.metrics.update({**world_model_metrics, **actor_critic_metrics})
        self.metrics.maybe_flush(step)

        if step % 25 == 0:
            logging.info("Gradient steps performed: %d", step)
