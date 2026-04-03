from typing import TypedDict

from torch import Tensor


class WorldModelInput(TypedDict):
    """World model input arguments.

    Represents a collected batch of environment transitions.
    """

    # (batch, sequence, channel, height, width)
    observations: Tensor

    # (batch, sequence, action_size)
    actions: Tensor

    # (batch, sequence)
    rewards: Tensor

    # (batch, sequence)
    dones: Tensor


class ObservedOutput(TypedDict):
    # (batch, sequence, channel, height, width)
    reconstructed_observations: Tensor

    # (batch, sequence, full_state_size)
    full_states: Tensor

    # (batch, sequence, recurrent_size)
    recurrent_states: Tensor

    # (batch, sequence, categorical, class)
    posterior_log_probs: Tensor

    # (batch, sequence, categorical, class)
    prior_log_probs: Tensor

    # (batch, sequence, bins)
    predicted_reward_logits: Tensor

    # (batch, sequence, 1)
    predicted_continue_logits: Tensor


class DreamOutput(TypedDict):
    # (batch * rollouts, sequence + 1, full_state_size)
    full_states: Tensor

    # (batch * rollouts, sequence + 1, action_size)
    actions: Tensor

    # (batch * rollouts, sequence + 1, action_size)
    action_probs: Tensor

    # (batch * rollouts, sequence, bins)
    predicted_reward_logits: Tensor

    # (batch * rollouts, sequence, 1)
    predicted_continue_logits: Tensor
