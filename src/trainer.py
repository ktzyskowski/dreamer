import logging

import torch
import mlflow

from src.loss.actor_critic_loss import ActorCriticLoss
from src.loss.world_model_loss import WorldModelLoss
from src.models.actor import DiscreteActor
from src.models.critic import Critic
from src.models.world_model import WorldModel
from src.util.buffer import ReplayBuffer
from src.util.env import EnvironmentManager
from src.util.functions import get_device


class Trainer:
    def __init__(
        self,
        env: EnvironmentManager,
        world_model: WorldModel,
        actor: DiscreteActor,
        critic: Critic,
        replay_buffer: ReplayBuffer,
        config,
    ):
        self.env = env
        self.replay_buffer = replay_buffer

        self.gradient_step_counter = 0

        # =================================================
        # hyperparameters
        self.device = get_device(priority=config.device)
        logging.info("Using device %s", self.device)

        self.warmup_steps = config.warmup_steps
        self.replay_ratio = config.replay_ratio
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length

        # =================================================
        # models
        self.world_model = world_model.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # =================================================
        # optimizers
        self.world_model_optimizer = torch.optim.Adam(
            world_model.parameters(),
            lr=config.world_model.learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config.actor.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=config.critic.learning_rate,
        )

        # =================================================
        # loss functions
        self.world_model_loss = WorldModelLoss(config)
        self.actor_critic_loss = ActorCriticLoss(config)

    def _batch_to_device(self, batch):
        """Move a batch to the trainer device, and convert any inputs to tensors."""
        result = {}

        for k, v in batch.items():
            t = torch.tensor(v, dtype=torch.float32).to(self.device)
            if k == "observations":
                t = t / 255.0
            result[k] = t

        return result
        # batch_tensors = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}
        # return batch_tensors

    def train(self, env, n_steps=256):
        """Train the world model and actor/critic networks."""

        episode_reward = 0.0
        episode_length = 0
        recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
        observation = env.reset()
        for step in range(n_steps):
            with torch.no_grad():
                encoded_observation = self.world_model.encoder(observation.to(self.device))
                full_state, _ = self.world_model.get_full_state(encoded_observation, recurrent_state)
                action, _ = self.actor(full_state)
            action_idx = action.argmax(dim=-1).cpu().item()

            next_observation, reward, done = env.step(action_idx)
            self.replay_buffer.add(observation, action, reward, done)

            episode_reward += reward
            episode_length += 1

            if done:
                recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
                observation = env.reset()

                mlflow.log_metrics(
                    {
                        "env/episode_reward": episode_reward,
                        "env/episode_length": episode_length,
                    },
                    step=step,
                )
                episode_reward = 0.0
                episode_length = 0
            else:
                recurrent_state = self.world_model.get_next_recurrent_state(full_state, action, recurrent_state)
                observation = next_observation

            # perform gradient step
            if step >= self.warmup_steps and step % self.replay_ratio == 0:
                self.gradient_step()

    def gradient_step(self):
        batch = self.replay_buffer.sample(batch_size=self.batch_size, sequence_length=self.sequence_length)
        batch = self._batch_to_device(batch)

        # ======================================================= #

        # train world model first on observed rollouts
        self.world_model_optimizer.zero_grad()
        observed_output = self.world_model.observe(batch)
        world_model_loss = self.world_model_loss(batch, observed_output)
        world_model_loss.backward()
        self.world_model_optimizer.step()

        # ======================================================= #

        # generate dream rollouts to train actor/critic
        recurrent_states = observed_output["recurrent_states"].detach()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        self.world_model.requires_grad_(False)
        dream_output = self.world_model.dream(batch, recurrent_states, self.actor)
        self.world_model.requires_grad_(True)

        actor_critic_loss = self.actor_critic_loss(dream_output, self.critic)
        actor_critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # ======================================================= #

        # log metrics
        mlflow.log_metrics(
            {
                **self.world_model_loss.metrics,
                **self.actor_critic_loss.metrics,
            },
            step=self.gradient_step_counter,
        )

        self.gradient_step_counter += 1
        if self.gradient_step_counter % 25 == 0:
            logging.info("Gradient steps performed: %d", self.gradient_step_counter)
