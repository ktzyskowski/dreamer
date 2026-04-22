import torch
import torch.nn as nn

from src.nets.mlp import MultiLayerPerceptron
from src.rl.agent import Agent
from src.rl.world_model import WorldModel, get_full_state
from src.util.probability import policy_distribution


class Dreamer(nn.Module):
    """Top-level Dreamer model.

    Owns the world model, agent (actor + critic), encoder/decoder, and the
    reward/continue prediction heads. Exposes three entry points:

      observe(batch): run real transitions through the world model and heads.
        Output dict is consumed by WorldModelLoss.
      dream(full_states, recurrent_states): roll the world model forward in
        latent space using the agent's actor. Output dict is consumed by
        ActorCriticLoss.
      act(observation, recurrent_state): single env step for the Collector.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        world_model: WorldModel,
        agent: Agent,
        reward_predictor: MultiLayerPerceptron,
        continue_predictor: MultiLayerPerceptron,
        dream_horizon: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.world_model = world_model
        self.agent = agent
        self.reward_predictor = reward_predictor
        self.continue_predictor = continue_predictor

        self.dream_horizon = dream_horizon

        # zero-init last layer of reward predictor to avoid hallucinating
        # rewards before any training has happened
        nn.init.zeros_(self.reward_predictor.net[-1].weight)  # type: ignore
        nn.init.zeros_(self.reward_predictor.net[-1].bias)  # type: ignore

    def world_model_parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.world_model.parameters())
            + list(self.reward_predictor.parameters())
            + list(self.continue_predictor.parameters())
        )

    def freeze_world_model(self):
        for p in self.world_model_parameters():
            p.requires_grad_(False)

    def unfreeze_world_model(self):
        for p in self.world_model_parameters():
            p.requires_grad_(True)

    def actor_parameters(self):
        return list(self.agent.actor.parameters())

    def critic_parameters(self):
        return list(self.agent.critic.fast.parameters())

    def observe(self, batch: dict) -> dict:
        """Run real transitions through the encoder, world model, and heads.

        Args:
            batch (dict): transition batch with keys "observations", "actions",
                "dones" of shape (B, T, *).

        Returns:
            dict with the keys WorldModelLoss expects: "posterior_logits",
            "prior_logits", "full_states", "recurrent_states",
            "reconstructed_observations", "predicted_reward_logits",
            "predicted_continue_logits".
        """
        encoded_observations = self.encoder(batch["observations"])
        output = self.world_model(encoded_observations, batch["actions"], batch["dones"])
        full_states = output["full_states"]
        output["reconstructed_observations"] = self.decoder(full_states)
        output["predicted_reward_logits"] = self.reward_predictor(full_states)
        output["predicted_continue_logits"] = self.continue_predictor(full_states)
        return output

    def dream(self, full_states: torch.Tensor, recurrent_states: torch.Tensor) -> dict:
        """Generate imagined rollouts seeded from every observed model state.

        Each of the B*T states from the observe phase seeds its own rollout.
        From there we step the world model forward using the prior over
        latents and the agent's actor.

        Args:
            full_states (B, T, full_state_size): full states from observe.
                Caller is expected to detach to prevent gradient leakage into
                the world model during actor-critic training.
            recurrent_states (B, T, recurrent_size): recurrent states from
                observe. Caller is expected to detach.

        Returns:
            dict with the keys ActorCriticLoss expects: "full_states",
            "actions", "action_logits", "predicted_reward_logits",
            "predicted_continue_logits". Sequence length is dream_horizon + 1;
            reward/continue logits cover the dreamed steps only (length
            dream_horizon).
        """
        # flatten (batch, time) into one batch dim for the rollout
        B, T = full_states.shape[:2]
        full_state = full_states.flatten(0, 1)
        recurrent_state = recurrent_states.flatten(0, 1)
        # full_state layout is (recurrent | latent); slice the latent back out
        latent_state = full_state[..., self.world_model.recurrent_size :]

        action_logits = self.agent.actor(full_state)
        action = policy_distribution(action_logits).sample()

        out_full_states = [full_state]
        out_action_logits = [action_logits]
        out_actions = [action]

        for _ in range(self.dream_horizon):
            recurrent_state = self.world_model.step(latent_state, recurrent_state, action)
            latent_state = self.world_model.get_prior_latent_state(recurrent_state)
            full_state = get_full_state(latent_state, recurrent_state)
            action_logits = self.agent.actor(full_state)
            action = policy_distribution(action_logits).sample()

            out_full_states.append(full_state)
            out_action_logits.append(action_logits)
            out_actions.append(action)

        full_states = torch.stack(out_full_states, dim=1).unflatten(0, (B, T))
        action_logits = torch.stack(out_action_logits, dim=1).unflatten(0, (B, T))
        actions = torch.stack(out_actions, dim=1).unflatten(0, (B, T))

        # heads are evaluated on the dreamed steps only; the seed step at
        # index 0 came from real data and is not used as a target
        predicted_reward_logits = self.reward_predictor(full_states[:, :, 1:])
        predicted_continue_logits = self.continue_predictor(full_states[:, :, 1:])

        return {
            "full_states": full_states,
            "actions": actions,
            "action_logits": action_logits,
            "predicted_reward_logits": predicted_reward_logits,
            "predicted_continue_logits": predicted_continue_logits,
        }

    @torch.no_grad()
    def act(self, observation: torch.Tensor, recurrent_state: torch.Tensor, greedy: bool = False):
        """Single env step. Inputs and outputs are unbatched.

        Returns:
            action: one-hot action tensor of shape (action_size,), sampled
                stochastically from the policy.
            next_recurrent_state: tensor of shape (recurrent_size,).
        """
        # GRU cells require a leading batch dim; manage it internally so the
        # Collector can keep holding unbatched per-env state
        encoded = self.encoder(observation).unsqueeze(0)
        recurrent_state = recurrent_state.unsqueeze(0)

        latent_state = self.world_model.get_posterior_latent_state(encoded, recurrent_state)
        full_state = get_full_state(latent_state, recurrent_state)
        action_logits = self.agent.actor(full_state)
        if greedy:
            action = action_logits.argmax(dim=-1)
        else:
            action = policy_distribution(action_logits, uniform_mix=0.01).sample()

        next_recurrent_state = self.world_model.step(latent_state, recurrent_state, action)
        return action.squeeze(0), next_recurrent_state.squeeze(0)
