from typing import Type

import torch
import torch.nn as nn

from src.rl.multi_categorical import MultiCategorical
from src.nets.rnn import BlockDiagonalGRU
from src.nets.mlp import MultiLayerPerceptron


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

    def forward(
        self,
        encoded_observations: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ):
        batch_size, sequence_length = encoded_observations.shape[0], encoded_observations.shape[1]

        output = {
            "full_states": [],
            "recurrent_states": [],
            "posterior_log_probs": [],
        }

        # initial recurrent state is a zero tensor
        recurrent_state = torch.zeros((batch_size, self.recurrent_size), device=encoded_observations.device)

        # iterate through each time step to collect recurrent states, and posterior/prior log probs
        for t in range(sequence_length):
            encoded_observation = encoded_observations[:, t]
            action = actions[:, t]

            posterior = MultiCategorical(
                logits=self.posterior_net(torch.cat([recurrent_state, encoded_observation], dim=-1)),
                n_categoricals=self.n_categoricals,
                n_classes=self.n_classes,
            )
            latent_state = posterior.sample()
            full_state = torch.cat([recurrent_state, latent_state], dim=-1)
            next_recurrent_state = self.recurrent_net(torch.cat([full_state, action], dim=-1), recurrent_state)
            posterior_log_probs = posterior.log_probs

            output["full_states"].append(full_state)
            output["recurrent_states"].append(recurrent_state)
            output["posterior_log_probs"].append(posterior_log_probs)

            # reset recurrent state at episode boundaries
            not_done = (~dones[:, t].bool()).unsqueeze(-1)
            recurrent_state = next_recurrent_state * not_done

        # stack all model outputs in sequence dimension
        for key in output.keys():
            output[key] = torch.stack(output[key], dim=1)

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
