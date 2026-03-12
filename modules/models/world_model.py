import torch
from torch import nn

from modules.nn.decoder import Decoder
from modules.utils.discrete_latent import DiscreteLatent
from modules.nn.encoder import Encoder
from modules.nn.mlp import MultiLayerPerceptron
from modules.nn.rnn import BlockDiagonalGRU


class WorldModel(nn.Module):
    def __init__(self, observation_shape, action_size, config):
        super().__init__()
        self.recurrent_size = config.world_model.recurrent_size
        self.n_categoricals = config.world_model.n_categoricals
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
        # iterate through each time step in sequence to collect recurrent states,
        # and posterior/prior logits for latent state
        full_states, recurrent_states, posteriors, priors = [], [], [], []
        for t in range(sequence_length):
            full_state, next_recurrent_state, posterior_logits, prior_logits = (
                self.observed_step(observations[:, t], actions[:, t], recurrent_state)
            )
            
            full_states.append(full_state)
            recurrent_states.append(recurrent_state)
            posteriors.append(posterior_logits)
            priors.append(prior_logits)
            recurrent_state = next_recurrent_state

        return {
            **batch,
            "recurrent_states": torch.stack(recurrent_states, dim=1),
            "full_states": torch.stack(full_states, dim=1),
            "posterior_logits": torch.stack(posteriors, dim=1),
            "prior_logits": torch.stack(priors, dim=1),
        }

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
        """_summary_

        Args:
            observation (batch, *observation_shape): _description_
            action (batch, *action_shape): _description_
            recurrent_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # extract latent state logits through posterior net
        encoded_observation = self.encoder(observation)
        posterior_logits = self.posterior_net(
            torch.cat([encoded_observation, recurrent_state], dim=-1)
        )
        prior_logits = self.prior_net(recurrent_state)
        # use posterior logits to generate/sample a discrete state
        posterior_sample = DiscreteLatent(
            logits=posterior_logits,
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        ).sample()
        # pass through sequence model to get next recurrent state
        next_recurrent_state = self.recurrent_model(
            torch.cat((posterior_sample, recurrent_state, action), dim=-1)
        )
        full_state = torch.cat([recurrent_state, posterior_sample], dim=1)
        return full_state, next_recurrent_state, posterior_logits, prior_logits

    def imagined_step(self, action, recurrent_state):
        # same flow as observe_step(), except we do not have an observation,
        # so we must use the prior net to generate latent states
        prior_logits = self.prior_net(recurrent_state)
        prior_sample = DiscreteLatent(
            logits=prior_logits,
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        ).sample()
        next_recurrent_state = self.recurrent_model(
            torch.cat((prior_sample, recurrent_state, action), dim=-1)
        )
        full_state = torch.cat([recurrent_state, prior_sample], dim=1)
        return full_state, next_recurrent_state, prior_logits
