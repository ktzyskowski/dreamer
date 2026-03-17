import torch

from buffer import ReplayBuffer
from models.actor import DiscreteActor
from models.world_model import WorldModel
from src.nn.functions import get_device
from src.nn.losses import ActorLoss, WorldModelLoss


class Trainer:
    def __init__(self, world_model: WorldModel, actor: DiscreteActor, replay_buffer: ReplayBuffer, config):
        # self.config = config
        # =================================================
        # hyperparameters
        self.device = get_device(priority=config.device)
        self.warmup_steps = config.warmup_steps
        self.replay_ratio = config.replay_ratio

        # =================================================
        # models
        self.replay_buffer = replay_buffer
        self.world_model = world_model.to(self.device)
        self.actor = actor.to(self.device)

        # =================================================
        # optimizers
        self.world_model_optimizer = torch.optim.Muon(
            world_model.parameters(),
            lr=config.training.world_model_lr,
        )
        self.actor_optimizer = torch.optim.Muon(
            actor.parameters(),
            lr=config.training.actor_lr,
        )

        # =================================================
        # loss functions
        self.world_model_loss = WorldModelLoss(
            beta_posterior=0.1,
            beta_prior=1.0,
            beta_prediction=1.0,
            free_nats=1.0,
        )
        self.actor_loss = ActorLoss(
            gamma=config.training.gamma,
            entropy_coef=config.training.entropy_coef,
        )

    def _batch_to_device(self, batch):
        """Move a batch to the trainer device, and convert any inputs to tensors."""
        return {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}

    def train(self, env, n_steps=256):
        """Train the world model and actor/critic networks."""

        recurrent_state = torch.zeros(self.world_model.recurrent_size, device=self.device)
        observation = env.reset()
        for step in range(n_steps):
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

            # perform gradient step
            if step >= self.warmup_steps and step % replay_ratio == 0:
                self.gradient_step()

    def gradient_step(self):
        batch = self.replay_buffer.sample()


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
