from typing import Type

import torch
import torch.nn as nn

from src.nets.mlp import MultiLayerPerceptron
from src.nets.rnn import BlockDiagonalGRU
from src.rl.multi_categorical import MultiCategorical


class WorldModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        recurrent_size: int,
        action_size: int,
        hidden_sizes: list[int],
        n_categoricals: int,
        n_classes: int,
        activation: Type[nn.Module],
        n_blocks: int,
    ):
        """Construct a new world model.

        Args:
            input_size (int): size of encoded input observation.
            recurrent_size (int): size of recurrent state.
            action_size (int): number of actions.
            hidden_sizes (list[int]): sizes of hidden layers used in MLPs.
            n_categoricals (int): number of latent categoricals.
            n_classes (int): number of classes per latent categorical.
            activation (Type[nn.Module]): activation function used in MLPs.
            n_blocks (int): number of blocks in the recurrent GRU.
        """
        super().__init__()
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.action_size = action_size
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.latent_size = n_categoricals * n_classes
        self.full_state_size = self.latent_size + self.recurrent_size

        # z ~ p(z | h, x)
        self.posterior_net = MultiLayerPerceptron(
            input_dim=self.recurrent_size + self.input_size,
            hidden_dims=hidden_sizes,
            output_dim=self.latent_size,
            activation=activation,
            output_activation=None,
        )

        # z ~ p(z | h)
        self.prior_net = MultiLayerPerceptron(
            input_dim=self.recurrent_size,
            hidden_dims=hidden_sizes,
            output_dim=self.latent_size,
            activation=activation,
            output_activation=None,
        )

        # h' = f(z, h, a)
        self.recurrent_net = BlockDiagonalGRU(
            input_size=self.full_state_size + action_size,
            recurrent_size=recurrent_size,
            n_blocks=n_blocks,
        )

    def step(self):
        pass

    def forward(
        self,
        encoded_observations: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ):
        """_summary_

        Args:
            encoded_observations (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            dones (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        batch_size, sequence_length = (
            encoded_observations.shape[0],
            encoded_observations.shape[1],
        )

        output = {
            "full_states": [],
            "recurrent_states": [],
            "posterior_log_probs": [],
        }

        # initial recurrent state is a zero tensor
        recurrent_state = torch.zeros(
            (batch_size, self.recurrent_size), device=encoded_observations.device
        )

        # iterate through each time step to collect recurrent states, and posterior/prior log probs
        for t in range(sequence_length):
            posterior = MultiCategorical(
                logits=self.posterior_net(
                    torch.cat([recurrent_state, encoded_observations[:, t]], dim=-1)
                ),
                n_categoricals=self.n_categoricals,
                n_classes=self.n_classes,
            )
            latent_state = posterior.sample()
            full_state = torch.cat([recurrent_state, latent_state], dim=-1)
            next_recurrent_state = self.recurrent_net(
                torch.cat([full_state, actions[:, t]], dim=-1), recurrent_state
            )
            posterior_log_probs = posterior.log_probs

            output["full_states"].append(full_state)
            output["recurrent_states"].append(recurrent_state)
            output["posterior_log_probs"].append(posterior_log_probs)

            # reset recurrent state at episode boundaries
            not_done = (~dones[:, t].bool()).unsqueeze(-1)
            recurrent_state = next_recurrent_state * not_done

        # stack all model outputs in sequence dimension
        output = {key: torch.stack(output[key], dim=1) for key in output.keys()}

        # prior is not used for sampling latent states, just log-probs;
        # we can compute them all at once outside the loop once we have
        # the recurrent states for each timestep.
        prior = MultiCategorical(
            logits=self.prior_net(output["recurrent_states"]),
            n_categoricals=self.n_categoricals,
            n_classes=self.n_classes,
        )
        prior_log_probs = prior.log_probs
        output["prior_log_probs"] = prior_log_probs

        return output


# from typing import Optional
# import typing

# import torch
# from torch import Tensor, nn

# from data.data import DreamOutput, ObservedOutput, WorldModelInput
# from src.nets.decoder import Decoder, MLPDecoder
# from src.nets.encoder import Encoder, MLPEncoder
# from src.nets.mlp import MultiLayerPerceptron
# from src.nets.rnn import BlockDiagonalGRU
# from rl.multi_categorical import MultiCategorical


# class WorldModel(nn.Module):
#     """The main world model class."""

#     def __init__(self, observation_shape, action_size, config):
#         super().__init__()
#         # hyperparameters
#         self.dream_horizon = config.world_model.dream_horizon
#         self.n_dreams = config.world_model.n_dreams

#         # shapes/sizes
#         self.recurrent_size = config.world_model.recurrent_size
#         self.n_categoricals = config.world_model.n_categoricals
#         self.n_classes = config.world_model.n_classes
#         self.latent_size = self.n_categoricals * self.n_classes
#         self.full_state_size = self.latent_size + self.recurrent_size

#         # subnetworks
#         if len(observation_shape) == 1:
#             self.encoder = MLPEncoder(
#                 input_size=observation_shape[0],
#                 output_size=self.latent_size,
#                 hidden_dims=config.world_model.encoder.hidden_dims,
#             )
#             self.decoder = MLPDecoder(
#                 input_dim=self.full_state_size,
#                 output_size=observation_shape[0],
#                 hidden_dims=config.world_model.encoder.hidden_dims,
#             )
#         else:
#             self.encoder = Encoder(
#                 observation_shape=observation_shape,
#                 output_size=self.latent_size,
#                 kernel_size=config.world_model.encoder.kernel_size,
#                 stride=config.world_model.encoder.stride,
#                 padding=config.world_model.encoder.padding,
#             )
#             self.decoder = Decoder(
#                 observation_shape=observation_shape,
#                 input_dim=self.full_state_size,
#                 kernel_size=config.world_model.encoder.kernel_size,
#                 stride=config.world_model.encoder.stride,
#                 padding=config.world_model.encoder.padding,
#             )
#         self.posterior_net = MultiLayerPerceptron(
#             input_dim=self.full_state_size,
#             hidden_dims=config.world_model.posterior_net.hidden_dims,
#             output_dim=self.latent_size,
#         )
#         self.prior_net = MultiLayerPerceptron(
#             input_dim=self.recurrent_size,
#             hidden_dims=config.world_model.prior_net.hidden_dims,
#             output_dim=self.latent_size,
#         )
#         self.reward_predictor = MultiLayerPerceptron(
#             input_dim=self.full_state_size,
#             hidden_dims=config.world_model.reward_predictor.hidden_dims,
#             output_dim=config.two_hot.n_bins,
#         )
#         self.continue_predictor = MultiLayerPerceptron(
#             input_dim=self.full_state_size,
#             hidden_dims=config.world_model.continue_predictor.hidden_dims,
#             output_dim=1,
#         )
#         self.recurrent_model = BlockDiagonalGRU(
#             input_size=self.full_state_size + action_size,
#             recurrent_size=self.recurrent_size,
#             n_blocks=config.world_model.recurrent_model.n_blocks,
#         )

#         # initialize weights of output layer in reward predictor to be zero,
#         # this is done to avoid hallucinating rewards early in training
#         nn.init.zeros_(self.reward_predictor.net[-1].weight)  # type: ignore
#         nn.init.zeros_(self.reward_predictor.net[-1].bias)  # type: ignore

#     @property
#     def device(self):
#         """Get the device where the parameters of the world model reside."""
#         device = next(self.parameters()).device
#         return device

#     def get_full_state(self, encoded_observation: Optional[Tensor], recurrent_state: Tensor):
#         """Get the full model state and latent log probabilities, given an encoded observation and recurrent state.

#         The encoded observation is used to condition the posterior distribution, from which a sample is collected
#         and concatenated with the given recurrent state to produce a model state.

#         If an observation is not provided, then the prior distribution is used instead to generate a sample.
#         """
#         if encoded_observation is not None:
#             # use posterior
#             distribution = MultiCategorical(
#                 logits=self.posterior_net(torch.cat([recurrent_state, encoded_observation], dim=-1)),
#                 n_categoricals=self.n_categoricals,
#                 n_classes=self.n_classes,
#             )
#         else:
#             # use prior
#             distribution = MultiCategorical(
#                 logits=self.prior_net(recurrent_state),
#                 n_categoricals=self.n_categoricals,
#                 n_classes=self.n_classes,
#             )
#         sample = distribution.sample()
#         log_probs = distribution.log_probs
#         full_state = torch.cat([recurrent_state, sample], dim=-1)
#         return full_state, log_probs

#     def get_next_recurrent_state(self, full_state, action, recurrent_state):
#         """Compute the next recurrent state.

#         The full state, action, and recurrent state are all concatenated and passed to the underlying
#         recurrent model.
#         """
#         next_recurrent_state = self.recurrent_model(torch.cat([full_state, action], dim=-1), recurrent_state)
#         return next_recurrent_state

#     def observe(self, batch: WorldModelInput) -> ObservedOutput:
#         """Observe the given batch of environment transitions.

#         Each trajectory is passed through the world model to generate model states
#         and log probabilities as needed for loss calculations.
#         """

#         observations = batch["observations"]
#         # batch encode observations outside of loop
#         encoded_observations = self.encoder(observations)
#         actions = batch["actions"]
#         dones = batch["dones"]
#         batch_size, sequence_length = observations.shape[0], observations.shape[1]

#         # initialize recurrent state to zero tensors
#         recurrent_state = torch.zeros((batch_size, self.recurrent_size), device=observations.device)

#         # storing model outputs in a dictionary to keep things neat,
#         # we will convert the lists to tensors at the end
#         observed_output = {
#             # reconstructed_observations=[],
#             "full_states": [],
#             "recurrent_states": [],
#             "posterior_log_probs": [],
#             "prior_log_probs": [],
#             # predicted_reward_logits=[],
#             # predicted_continue_logits=[],
#         }

#         # iterate through each time step to collect recurrent states, and posterior/prior log probs
#         for t in range(sequence_length):
#             full_state, next_recurrent_state, posterior_log_probs, prior_log_probs = self.observed_step(
#                 encoded_observations[:, t], actions[:, t], recurrent_state
#             )
#             # reconstructed_observation = self.decoder(full_state)
#             # predicted_reward_logits = self.reward_predictor(full_state)
#             # predicted_continue_logits = self.continue_predictor(full_state)
#             # observed_output["reconstructed_observations"].append(reconstructed_observation)
#             # observed_output["predicted_reward_logits"].append(predicted_reward_logits)
#             # observed_output["predicted_continue_logits"].append(predicted_continue_logits)
#             observed_output["full_states"].append(full_state)
#             observed_output["recurrent_states"].append(recurrent_state)
#             observed_output["posterior_log_probs"].append(posterior_log_probs)
#             observed_output["prior_log_probs"].append(prior_log_probs)
#             # reset recurrent state for sequences that just crossed an episode boundary
#             not_done = (~dones[:, t].bool()).unsqueeze(-1)
#             recurrent_state = next_recurrent_state * not_done

#         # stack all model outputs in sequence dimension
#         for key in observed_output.keys():
#             observed_output[key] = torch.stack(observed_output[key], dim=1)  # type: ignore

#         # batch process full states outside of loop
#         observed_output["reconstructed_observations"] = self.decoder(observed_output["full_states"])
#         observed_output["predicted_continue_logits"] = self.continue_predictor(observed_output["full_states"])
#         observed_output["predicted_reward_logits"] = self.reward_predictor(observed_output["full_states"])

#         return observed_output

#     def observed_step(self, encoded_observation: Tensor, action: Tensor, recurrent_state: Tensor):
#         """Take an observed step through the world model.

#         Log-probabilities are computed for both the posterior and prior latent distributions, even though
#         only the posterior distribution is used to generate samples for the model state. The prior log-probs
#         are used later during loss calculation for the world model.
#         """
#         full_state, posterior_log_probs = self.get_full_state(
#             encoded_observation=encoded_observation,
#             recurrent_state=recurrent_state,
#         )
#         _, prior_log_probs = self.get_full_state(
#             encoded_observation=None,
#             recurrent_state=recurrent_state,
#         )
#         recurrent_state = self.get_next_recurrent_state(full_state, action, recurrent_state)
#         return full_state, recurrent_state, posterior_log_probs, prior_log_probs

#     def dream(self, batch: WorldModelInput, recurrent_states: Tensor, actor: nn.Module) -> DreamOutput:
#         """Generate dream rollouts from the given batch of transitions.

#         The recurrent states that were generated during the observation phase are used
#         as initial recurrent context for the world model during each rollout.
#         """

#         observations = batch["observations"]
#         encoded_observations = self.encoder(observations)
#         sequence_length = observations.shape[1]

#         # randomly sample starting timesteps instead of using all
#         starting_timesteps = torch.randperm(sequence_length, device=recurrent_states.device)[: self.n_dreams]

#         # select starting states and flatten batch & sequences into one batch dimension
#         flattened_observations = encoded_observations[:, starting_timesteps].flatten(0, 1)
#         flattened_recurrent_states = recurrent_states[:, starting_timesteps].flatten(0, 1)

#         rollouts = self.dream_rollout(flattened_observations, flattened_recurrent_states, actor)
#         rollouts = typing.cast(DreamOutput, rollouts)

#         return rollouts

#     def dream_rollout(self, encoded_observation: Tensor, recurrent_state: Tensor, actor: nn.Module):
#         # encoded_observation:  (batch, latent_size)
#         # recurrent_state:      (batch, recurrent_size)

#         # our initial full state is created from the posterior distribution,
#         # conditioned on the initial observation
#         full_state, _ = self.get_full_state(encoded_observation=encoded_observation, recurrent_state=recurrent_state)
#         action, action_probs = actor(full_state)

#         rollout_output = {
#             "full_states": [full_state],
#             "actions": [action],
#             "action_probs": [action_probs],
#             # "predicted_reward_logits": [],
#             # "predicted_continue_logits": [],
#         }

#         # from here on out, we're sailing through a dream!
#         for _ in range(self.dream_horizon):
#             full_state, recurrent_state, _ = self.dream_step(action, recurrent_state)
#             action, action_probs = actor(full_state)
#             # predicted_reward_logits = self.reward_predictor(full_state)
#             # predicted_continue_logits = self.continue_predictor(full_state)
#             rollout_output["full_states"].append(full_state)
#             rollout_output["actions"].append(action)
#             rollout_output["action_probs"].append(action_probs)
#             # rollout_output["predicted_reward_logits"].append(predicted_reward_logits)
#             # rollout_output["predicted_continue_logits"].append(predicted_continue_logits)

#         # stack all rollout outputs in sequence dimension
#         for key in rollout_output.keys():
#             rollout_output[key] = torch.stack(rollout_output[key], dim=1)  # type: ignore

#         rollout_output["predicted_reward_logits"] = self.reward_predictor(rollout_output["full_states"][:, 1:])
#         rollout_output["predicted_continue_logits"] = self.continue_predictor(rollout_output["full_states"][:, 1:])

#         # rollout_output
#         # ==============
#         # full_states:                  (batch, dream_sequence + 1, full_state_size)
#         # actions:                      (batch, dream_sequence + 1, action_size)
#         # action_probs:                 (batch, dream_sequence + 1, action_size)
#         # predicted_reward_logits:      (batch, dream_sequence, bins)
#         # predicted_continue_logits:    (batch, dream_sequence, 1)
#         return rollout_output

#     def dream_step(self, action, recurrent_state):
#         """Take an dream step through the world model.

#         Because we do not have an observation, we only utilize the prior distribution to create our model state.
#         """
#         full_state, prior_log_probs = self.get_full_state(encoded_observation=None, recurrent_state=recurrent_state)
#         next_recurrent_state = self.get_next_recurrent_state(full_state, action, recurrent_state)
#         return full_state, next_recurrent_state, prior_log_probs
