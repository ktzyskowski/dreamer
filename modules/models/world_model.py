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

    def step(self, observation, action, recurrent_state):
        """Process one step in the world model.

        The current observation is encoded, and when combined
        with the action and prior recurrent state, are used
        to generate the next recurrent state.

        Args:
            observation (*observation_shape):
            action (*action_shape):
            recurrent_state (recurrent_dim):
        Returns:
            recurrent_state (recurrent_dim):
            discrete_state (discrete_dim):
        """
        # extract latent state logits through posterior net
        encoded_observation = self.encoder(observation)
        posterior_logits = self.posterior_net(
            torch.cat([encoded_observation, recurrent_state], dim=-1)
        )
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
        return posterior_sample, next_recurrent_state

    def observe(self, batch):
        pass

    def imagine(self, batch):
        pass
