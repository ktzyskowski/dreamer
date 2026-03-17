import torch
from torch import nn

from src.nets.decoder import Decoder
from src.nets.discrete_latent import DiscreteLatent
from src.nets.encoder import Encoder
from src.nets.mlp import MultiLayerPerceptron
from src.nets.rnn import BlockDiagonalGRU
from src.nets.functions import symlog
from src.nets.two_hot import TwoHot


def _condense_rollouts(list_of_rollouts):
    """Condense a list of dicts of rollout tensors into a single dict of tensors."""

    assert len(list_of_rollouts) > 0
    keys = list_of_rollouts[0].keys()

    rollouts = {}
    for key in keys:
        tensors = [rollout[key] for rollout in list_of_rollouts]
        rollouts[key] = torch.stack(tensors, dim=0)

    return rollouts


class WorldModel(nn.Module):
    def __init__(self, observation_shape, action_size, config):
        super().__init__()
        self.dream_horizon = config.world_model.dream_horizon
        self.recurrent_size = config.world_model.recurrent_size
        self.n_categoricals = config.world_model.n_categoricals
        self.n_classes = config.world_model.n_classes
        self.latent_size = self.n_categoricals * self.n_classes
        self.full_state_size = self.latent_size + self.recurrent_size
        self.two_hot = TwoHot(
            config.world_model.twohot.low,
            config.world_model.twohot.high,
            config.world_model.twohot.n_bins,
        )
        self.encoder = Encoder(
            observation_shape=observation_shape,
            output_size=self.latent_size,
            kernel_size=config.world_model.encoder.kernel_size,
            stride=config.world_model.encoder.stride,
            padding=config.world_model.encoder.padding,
        )
        self.decoder = Decoder(
            observation_shape=observation_shape,
            input_dim=self.latent_size,
            kernel_size=config.world_model.encoder.kernel_size,
            stride=config.world_model.encoder.stride,
            padding=config.world_model.encoder.padding,
        )
        self.posterior_net = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=config.world_model.posterior_net.hidden_layers,
            output_dim=self.latent_size,
        )
        self.prior_net = MultiLayerPerceptron(
            input_dim=self.recurrent_size,
            hidden_dims=config.world_model.prior_net.hidden_layers,
            output_dim=self.latent_size,
        )
        self.reward_predictor = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=config.world_model.reward_predictor.hidden_layers,
            output_dim=config.world_model.twohot.n_bins,
        )
        self.continue_predictor = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=config.world_model.continue_predictor.hidden_layers,
            output_dim=1,
        )
        self.recurrent_model = BlockDiagonalGRU(
            input_size=self.full_state_size + action_size,
            recurrent_size=self.recurrent_size,
            n_blocks=config.world_model.recurrent_model.n_blocks,
        )

        # initialize weights of output layer in reward predictor to be zero,
        # this is done to avoid hallucinating rewards early in training
        nn.init.zeros_(self.reward_predictor.net[-1].weight)  # type: ignore
        nn.init.zeros_(self.reward_predictor.net[-1].bias)  # type: ignore

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def get_full_state(self, observation, recurrent_state):
        posterior = DiscreteLatent(
            logits=self.posterior_net(torch.cat([recurrent_state, self.encoder(observation)], dim=-1)),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        full_state = torch.cat([recurrent_state, posterior.sample()], dim=-1)
        return full_state

    def get_next_recurrent_state(self, full_state, action, recurrent_state):
        next_recurrent_state = self.recurrent_model(torch.cat([full_state, action], dim=-1), recurrent_state)
        return next_recurrent_state

    def observe(self, batch):
        """Observe the given batch of environment transitions.

        Each trajectory is passed through the world model to generate model states
        and log probabilities as needed for loss calculations.
        """
        # batch
        # =====
        # observations:     (batch, sequence, channel, height, width)
        # actions:          (batch, sequence, action_size)
        # rewards:          (batch, sequence)
        # dones:            (batch, sequence)

        observations = batch["observations"]
        actions = batch["actions"]
        batch_size, sequence_length = observations.shape[0], observations.shape[1]

        # initialize recurrent state
        recurrent_state = torch.zeros((batch_size, self.recurrent_size), device=observations.device)

        # storing model outputs in a dictionary keeps things neat, we will
        # combine with original batch dictionary when returning
        model_state = {
            "reconstructed_observations": [],
            "full_states": [],
            "recurrent_states": [],
            "posterior_log_probs": [],
            "prior_log_probs": [],
            "predicted_reward_logits": [],
            "predicted_continue_logits": [],
        }

        # iterate through each time step to collect recurrent states, and posterior/prior log probs
        for t in range(sequence_length):
            full_state, next_recurrent_state, posterior_log_probs, prior_log_probs = self.observed_step(
                observations[:, t], actions[:, t], recurrent_state
            )
            reconstructed_observation = self.decoder(full_state)
            predicted_reward_logits = self.reward_predictor(full_state)
            predicted_continue_logits = self.continue_predictor(full_state)
            model_state["reconstructed_observations"].append(reconstructed_observation)
            model_state["full_states"].append(full_state)
            model_state["recurrent_states"].append(recurrent_state)
            model_state["posterior_log_probs"].append(posterior_log_probs)
            model_state["prior_log_probs"].append(prior_log_probs)
            model_state["predicted_reward_logits"].append(predicted_reward_logits)
            model_state["predicted_continue_logits"].append(predicted_continue_logits)
            recurrent_state = next_recurrent_state

        # stack all model outputs in sequence dimension
        for key in model_state.keys():
            model_state[key] = torch.stack(model_state[key], dim=1)  # type: ignore

        # two-hot encode symlog rewards for cross-entropy loss
        model_state["rewards_symlog_two_hot"] = self.two_hot.encode(symlog(batch["rewards"]))  # type: ignore

        # model_state
        # ===========
        # reconstructed_observations:   (batch, sequence, channel, height, width)
        # full_states:                  (batch, sequence, full_state_size)
        # recurrent_states:             (batch, sequence, recurrent_size)
        # posterior_log_probs:          (batch, sequence, categorical, class)
        # prior_log_probs:              (batch, sequence, categorical, class)
        # predicted_reward_logits:      (batch, sequence, bins)
        # predicted_continue_logits:    (batch, sequence, 1)
        # rewards_symlog_two_hot:       (batch, sequence, bins)
        return {**batch, **model_state}

    def observed_step(self, observation, action, recurrent_state=None):
        """Take an observed step through the world model.

        The recurrent state is routed through both the posterior and prior net, even though the full state
        is constructed from the posterior sample only, since we'll need both log probs in order to calculate
        our world model loss.
        """
        # observation:          (*, channel, height, width)
        # action:               (*, action_size)
        # recurrent_state:      (*, recurrent_size)

        if recurrent_state is None:
            recurrent_state = torch.zeros(*observation.shape[:-3], self.recurrent_size, device=observation.device)
        posterior = DiscreteLatent(
            logits=self.posterior_net(torch.cat([recurrent_state, self.encoder(observation)], dim=-1)),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        prior = DiscreteLatent(
            logits=self.prior_net(recurrent_state),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        posterior_sample = posterior.sample()
        full_state = torch.cat([recurrent_state, posterior_sample], dim=-1)
        recurrent_state = self.recurrent_model(torch.cat([full_state, action], dim=-1), recurrent_state)

        # full_state:           (*, full_state_size)
        # next_recurrent_state: (*, recurrent_size)
        # posterior_log_probs:  (*, categorical, class)
        # prior_log_probs:      (*, categorical, class)
        return full_state, recurrent_state, posterior.log_probs, prior.log_probs

    def dream(self, batch, actor):
        """Dream

        Each trajectory is passed through the world model to generate model states
        and log probabilities as needed for loss calculations.
        """
        # batch
        # =====
        # observations:         (batch, sequence, channel, height, width)
        # actions:              (batch, sequence, action_size)
        # rewards:              (batch, sequence)
        # dones:                (batch, sequence)
        # recurrent_states:     (batch, sequence, recurrent_size)

        observations = batch["observations"]
        recurrent_states = batch["recurrent_states"]
        sequence_length = observations.shape[1]

        # iterate through each time step, we will start a new dream rollout
        # from each observation in the sequence
        rollouts = []
        for t in range(sequence_length):
            recurrent_state = recurrent_states[:, t]
            observation = observations[:, t]
            rollout = self.dream_rollout(observation, recurrent_state, actor)
            rollouts.append(rollout)

        # stack rollouts into single tensor per key
        rollouts = _condense_rollouts(rollouts)

        # flatten batch/rollout into single dimension
        for key in rollouts:
            rollouts[key] = rollouts[key].flatten(0, 1)

        return rollouts

    def dream_rollout(self, observation, recurrent_state, actor):
        # observation:      (batch, channel, height, width)
        # action:           (batch, action_size)
        # recurrent_state:  (batch, recurrent_size)

        # pass initial observation through posterior net to get first full state
        posterior = DiscreteLatent(
            logits=self.posterior_net(torch.cat([recurrent_state, self.encoder(observation)], dim=-1)),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        posterior_sample = posterior.sample()
        full_state = torch.cat([recurrent_state, posterior_sample], dim=-1)
        action, action_probs = actor(full_state)

        rollout_output = {
            "full_states": [full_state],
            "actions": [action],
            "action_probs": [action_probs],
            "predicted_reward_logits": [],
            "predicted_continue_logits": [],
        }

        # from here on out, we're sailing through a dream!
        for _ in range(self.dream_horizon):
            full_state, recurrent_state, _ = self.dream_step(action, recurrent_state)
            action, action_probs = actor(full_state)
            predicted_reward_logits = self.reward_predictor(full_state)
            predicted_continue_logits = self.continue_predictor(full_state)
            rollout_output["full_states"].append(full_state)
            rollout_output["actions"].append(action)
            rollout_output["action_probs"].append(action_probs)
            rollout_output["predicted_reward_logits"].append(predicted_reward_logits)
            rollout_output["predicted_continue_logits"].append(predicted_continue_logits)

        # stack all rollout outputs in sequence dimension
        for key in rollout_output.keys():
            rollout_output[key] = torch.stack(rollout_output[key], dim=1)  # type: ignore

        # rollout_output
        # ==============
        # full_states:                  (batch, dream_sequence + 1, full_state_size)
        # actions:                      (batch, dream_sequence + 1, action_size)
        # action_probs:                 (batch, dream_sequence + 1, action_size)
        # predicted_reward_logits:      (batch, dream_sequence, bins)
        # predicted_continue_logits:    (batch, dream_sequence, 1)
        return rollout_output

    def dream_step(self, action, recurrent_state):
        """Take an dream step through the world model.

        This method is similar to observed_step(), except no observation is provided.
        Because no observation is provided, the recurrent state is routed only through the prior net.
        """
        # action:               (*, action_size)
        # recurrent_state:      (*, recurrent_size)

        prior = DiscreteLatent(
            logits=self.prior_net(recurrent_state),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        prior_sample = prior.sample()
        full_state = torch.cat([recurrent_state, prior_sample], dim=-1)
        next_recurrent_state = self.recurrent_model(torch.cat([full_state, action], dim=-1), recurrent_state)

        # full_state:           (*, full_state_size)
        # next_recurrent_state: (*, recurrent_size)
        # prior_log_probs:      (*, categorical, class)
        return full_state, next_recurrent_state, prior.log_probs
