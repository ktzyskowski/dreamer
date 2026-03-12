import torch
from torch import nn

from modules.nn.decoder import Decoder
from modules.utils.discrete_latent import DiscreteLatent
from modules.nn.encoder import Encoder
from modules.nn.mlp import MultiLayerPerceptron
from modules.nn.rnn import BlockDiagonalGRU
from modules.utils.twohot import TwoHot


class WorldModel(nn.Module):
    def __init__(self, observation_shape, action_size, config):
        super().__init__()
        self.recurrent_size = config.world_model.recurrent_size
        self.n_categoricals = config.world_model.n_categoricals
        self.twohot = TwoHot(
            config.world_model.twohot.low,
            config.world_model.twohot.high,
            config.world_model.twohot.n_bins,
        )
        self.n_classes = config.world_model.n_classes
        latent_size = self.n_categoricals * self.n_classes
        full_state_size = latent_size + self.recurrent_size
        self.encoder = Encoder(
            observation_shape=observation_shape,
            output_dim=latent_size,
            kernel_size=config.world_model.encoder.kernel_size,
            stride=config.world_model.encoder.stride,
            padding=config.world_model.encoder.padding,
        )
        self.decoder = Decoder(
            observation_shape=observation_shape,
            input_dim=latent_size,
            kernel_size=config.world_model.encoder.kernel_size,
            stride=config.world_model.encoder.stride,
            padding=config.world_model.encoder.padding,
        )
        self.posterior_net = MultiLayerPerceptron(
            input_dim=latent_size + self.recurrent_size,
            hidden_dims=config.world_model.posterior_net.hidden_layers,
            output_dim=latent_size,
        )
        self.prior_net = MultiLayerPerceptron(
            input_dim=self.recurrent_size,
            hidden_dims=config.world_model.prior_net.hidden_layers,
            output_dim=latent_size,
        )
        self.reward_predictor = MultiLayerPerceptron(
            input_dim=full_state_size,
            hidden_dims=config.world_model.reward_predictor.hidden_layers,
            output_dim=config.twohot.n_bins,
        )
        self.continue_predictor = MultiLayerPerceptron(
            input_dim=full_state_size,
            hidden_dims=config.world_model.continue_predictor.hidden_layers,
            output_dim=1,
        )
        self.recurrent_model = BlockDiagonalGRU(
            input_dim=full_state_size + action_size,
            hidden_dim=self.recurrent_size,
            n_blocks=config.world_model.recurrent_model.n_blocks,
        )

        # initialize weights of output layer in reward predictor to be zero,
        # this is done to avoid hallucinating rewards early in training
        nn.init.zeros_(self.reward_predictor.net[-1].weight)  # type: ignore
        nn.init.zeros_(self.reward_predictor.net[-1].bias)  # type: ignore

    def observe(self, batch):
        # batch is a dict with keys:
        # - "observations": tensor[batch, sequence, *observation_shape]
        # - "actions": tensor[batch, sequence, *action_shape]
        observations = batch["observations"]
        actions = batch["actions"]
        batch_size, sequence_length = observations.shape[0], observations.shape[1]

        # initial recurrent state is zero
        recurrent_state = torch.zeros(
            (batch_size, self.recurrent_size), device=observations.device
        )

        # storing model outputs in a dictionary keeps things neat, we will
        # combine with original batch dictionary when returning
        model_state = {
            "reconstructed_observations": [],
            "full_states": [],
            "recurrent_states": [],
            "posterior_log_probs": [],
            "prior_log_probs": [],
            "predicted_rewards": [],
            "predicted_continues": [],
        }

        # iterate through each time step in sequence to collect recurrent states,
        # and posterior/prior logits for latent state
        for t in range(sequence_length):
            full_state, next_recurrent_state, posterior_logits, prior_logits = (
                self.observed_step(observations[:, t], actions[:, t], recurrent_state)
            )
            reconstructed_observation = self.decoder(full_state)
            predicted_reward = self.reward_predictor(full_state)
            predicted_continue = self.continue_predictor(full_state)
            model_state["reconstructed_observations"].append(reconstructed_observation)
            model_state["full_states"].append(full_state)
            model_state["recurrent_states"].append(recurrent_state)
            model_state["posterior_log_probs"].append(posterior_logits)
            model_state["prior_log_probs"].append(prior_logits)
            model_state["predicted_rewards"].append(predicted_reward)
            model_state["predicted_continues"].append(predicted_continue)
            recurrent_state = next_recurrent_state

        # stack all model outputs in sequence dimension
        for key in model_state.keys():
            model_state[key] = torch.stack(model_state[key], dim=1)  # type: ignore

        # two-hot encode rewards for cross-entropy loss
        model_state["rewards_twohot"] = self.twohot.encode(batch["rewards"])  # type: ignore

        return {**batch, **model_state}

    def imagine(self, batch, actor):
        imagination_horizon = 16

        # batch is a dict with keys:
        # - "observations": tensor[batch, sequence, *observation_shape]
        observations = batch["observations"]
        batch_size, sequence_length = observations.shape[0], observations.shape[1]
        for t in range(sequence_length):
            for h in range(imagination_horizon):
                pass

    def observed_step(self, observation, action, recurrent_state):
        # extract posterior/prior from recurrent state and observation
        encoded_observation = self.encoder(observation)
        posterior = DiscreteLatent(
            logits=self.posterior_net(
                torch.cat([recurrent_state, encoded_observation], dim=-1)
            ),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        prior = DiscreteLatent(
            logits=self.prior_net(recurrent_state),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        # rely on posterior when observation is given
        posterior_sample = posterior.sample()
        # combine latent posterior with recurrent state to produce full model state
        full_state = torch.cat([recurrent_state, posterior_sample], dim=1)
        # pass model state into sequence model with action to produce next recurrent state
        recurrent_state = self.recurrent_model(torch.cat((full_state, action), dim=-1))
        return full_state, recurrent_state, posterior.log_probs, prior.log_probs

    def imagined_step(self, action, recurrent_state):
        # similar to observed_step(), except no posterior net (no observation)
        prior = DiscreteLatent(
            logits=self.prior_net(recurrent_state),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        prior_sample = prior.sample()
        full_state = torch.cat([recurrent_state, prior_sample], dim=1)
        recurrent_state = self.recurrent_model(torch.cat([full_state, action], dim=-1))
        return full_state, recurrent_state, prior.log_probs
