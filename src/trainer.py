import torch

from loss.actor_critic_loss import ActorCriticLoss
from src.loss.actor_loss import ActorLoss
from src.loss.critic_loss import CriticLoss
from src.loss.world_model_loss import WorldModelLoss
from src.models.actor import DiscreteActor
from src.models.critic import Critic
from src.models.world_model import WorldModel
from src.util.buffer import ReplayBuffer
from src.util.env import EnvironmentManager
from src.util.functions import calculate_lambda_returns, get_device


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
        # =================================================
        # hyperparameters
        self.device = get_device(priority=config.device)
        self.warmup_steps = config.warmup_steps
        self.replay_ratio = config.replay_ratio

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
        self.world_model_loss = WorldModelLoss(
            beta_posterior=config.world_model_loss.beta_posterior,
            beta_prior=config.world_model_loss.beta_prior,
            beta_prediction=config.world_model_loss.beta_prediction,
            free_nats=config.world_model_loss.free_nats,
        )
        self.actor_critic_loss = ActorCriticLoss()

    def _batch_to_device(self, batch):
        """Move a batch to the trainer device, and convert any inputs to tensors."""
        return {
            k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in batch.items()
        }

    def train(self, env, n_steps=256):
        """Train the world model and actor/critic networks."""

        recurrent_state = torch.zeros(
            self.world_model.recurrent_size, device=self.device
        )
        observation = env.reset()
        for step in range(n_steps):
            with torch.no_grad():
                full_state = self.world_model.get_full_state(
                    observation.to(self.device), recurrent_state
                )
                action, _ = self.actor(full_state)
            action_idx = action.argmax(dim=-1).cpu().item()

            next_observation, reward, done = env.step(action_idx)
            self.replay_buffer.add(observation, action, reward, done)

            if done:
                recurrent_state = torch.zeros(
                    self.world_model.recurrent_size, device=self.device
                )
                observation = env.reset()
            else:
                recurrent_state = self.world_model.get_next_recurrent_state(
                    full_state, action, recurrent_state
                )
                observation = next_observation

            # perform gradient step
            if step >= self.warmup_steps and step % self.replay_ratio == 0:
                self.gradient_step()

    def gradient_step(self):
        batch = self.replay_buffer.sample(batch_size=16, sequence_length=64)
        batch = self._batch_to_device(batch)

        # ======================================================= #

        # train world model first on observed rollouts
        self.world_model_optimizer.zero_grad()
        awake_output = self.world_model.observe(batch)
        loss = self.world_model_loss(awake_output)
        loss.backward()
        self.world_model_optimizer.step()

        # ======================================================= #

        # generate dream rollouts to train actor/critic
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        self.world_model.requires_grad_(False)
        dream_output = self.world_model.dream(awake_output, actor=self.actor)
        self.world_model.requires_grad_(True)
        critic_value_logits, critic_values = self.critic(dream_output["full_states"])
        dream_output["critic_value_logits"] = critic_value_logits
        dream_output["critic_values"] = critic_values
        dream_output["rewards"] = self.world_model.two_hot.decode(
            dream_output["predicted_reward_logits"]
        )
        actor_critic_loss = self.actor_critic_loss(dream_output)
        actor_critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


# def train_world_model(self, batch):
#     """Run observe on a batch and backprop the world model loss.

#     Returns:
#         observed: the full observed batch (model outputs merged with original batch)
#         loss: scalar loss value
#     """
#     self.world_model_optimizer.zero_grad()
#     observed = self.world_model.observe(batch)
#     loss = self.world_model_loss(observed)
#     loss.backward()
#     self.world_model_optimizer.step()
#     return observed, loss.item()

# def train_actor(self, observed_batch):
#     """Dream from the observed batch and backprop the actor loss.

#     The world model is frozen during the dream so that gradients accumulate
#     only in actor parameters. Gradients still flow through the dream states
#     into the actor, giving a more accurate policy gradient signal than
#     re-evaluating the actor on detached states.

#     Returns:
#         loss: scalar loss value
#     """
#     self.actor_optimizer.zero_grad()

#     self.world_model.requires_grad_(False)
#     dream_rollout = self.world_model.dream(observed_batch, self.actor)
#     self.world_model.requires_grad_(True)

#     # dream_rollout
#     # ============
#     # full_states:               (N, dream_horizon + 1, full_state_size)
#     # actions:                   (N, dream_horizon + 1, action_size)
#     # action_probs:              (N, dream_horizon + 1, action_size)
#     # predicted_reward_logits:   (N, dream_horizon, bins)
#     # predicted_continue_logits: (N, dream_horizon, 1)

#     # Slice off the last state/action since there is no corresponding reward for it
#     actor_batch = {
#         "actions": dream_rollout["actions"][:, :-1],
#         "action_probs": dream_rollout["action_probs"][:, :-1],
#         "decoded_rewards": symexp(self.world_model.two_hot.decode(dream_rollout["predicted_reward_logits"])),
#         "predicted_continue_logits": dream_rollout["predicted_continue_logits"],
#     }

#     loss = self.actor_loss(actor_batch)
#     loss.backward()
#     self.actor_optimizer.step()
#     return loss.item()

# def train(self):
#     """Sample a batch, train the world model, then train the actor.

#     Returns:
#         dict of scalar loss values
#     """
#     batch = self.replay_buffer.sample(self.config.batch_size, self.config.sequence_length)
#     batch = self._batch_to_device(batch)

#     observed, wm_loss = self.train_world_model(batch)

#     # Detach all observed tensors so actor training cannot backprop into the world model
#     # via the observed sequence (gradients are only intended to flow through dream states)
#     observed_detached = {k: v.detach() if torch.is_tensor(v) else v for k, v in observed.items()}
#     actor_loss = self.train_actor(observed_detached)

#     return {"world_model_loss": wm_loss, "actor_loss": actor_loss}
