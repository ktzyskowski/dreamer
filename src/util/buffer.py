import numpy as np
from omegaconf import DictConfig
import torch


class ReplayBuffer:
    """Replay buffer for storing and sampling sequences of transitions."""

    def __init__(
        self,
        observation_shape: tuple,
        action_size: int,
        config: DictConfig,
    ):
        if config.replay_buffer.capacity <= 0:
            raise ValueError("Replay buffer capacity must be greater than zero.")

        self.capacity = config.replay_buffer.capacity
        self.is_full = False
        self.buffer_index = 0  # pointer to the next index to insert a transition
        # internal buffer arrays to store transitions
        dtype = np.dtype(config.replay_buffer.dtype)
        self.observations = np.zeros((self.capacity, *observation_shape), dtype=dtype)
        self.actions = np.zeros((self.capacity, action_size), dtype=dtype)
        self.rewards = np.zeros((self.capacity,), dtype=dtype)
        self.dones = np.zeros((self.capacity,), dtype=dtype)

    def __len__(self):
        """Get the current number of transitions stored in the buffer."""
        return self.capacity if self.is_full else self.buffer_index

    def add(self, observation, action, reward, done):
        if torch.is_tensor(observation):
            observation = observation.numpy(force=True)
        if torch.is_tensor(action):
            action = action.numpy(force=True)

        """Add a transition to the replay buffer, overwriting old transitions if capacity is exceeded."""
        self.observations[self.buffer_index] = observation
        self.actions[self.buffer_index] = action
        self.rewards[self.buffer_index] = reward
        self.dones[self.buffer_index] = done

        # increment buffer index and wrap around if we exceed capacity, overwriting old transitions
        self.buffer_index = (self.buffer_index + 1) % self.capacity
        if self.buffer_index == 0:
            self.is_full = True

    def sample(self, batch_size, sequence_length):
        """Sample a batch of sequences of transitions from the replay buffer."""
        if len(self) < sequence_length:
            raise ValueError("Not enough transitions in the buffer to sample a sequence of the requested length.")

        max_start_index = self.buffer_index - sequence_length
        start_index_pool = np.arange(0, max_start_index + 1)

        # if the buffer is full, we can also sample sequences that wrap around the end of the buffer,
        # because the buffer index has wrapped around and is now at the beginning of the buffer
        if self.is_full:
            start_index_pool = np.concatenate((start_index_pool, np.arange(self.buffer_index, self.capacity)))

        # filter out start indices where the sequence crosses an episode boundary.
        # a done=True at step t means the episode ended there; step t+1 belongs to a new episode.
        # so we check the first (sequence_length - 1) dones. if any are True, the sequence is invalid.
        if sequence_length > 1:
            start_index_pool = np.array(
                [
                    idx
                    for idx in start_index_pool
                    if not np.any(gather_sequence(self.dones, idx, sequence_length - 1).astype(bool))
                ]
            )

        if len(start_index_pool) < batch_size:
            raise ValueError("Not enough valid sequences (respecting episode boundaries) to fill the requested batch.")

        start_indices = np.random.choice(start_index_pool, size=batch_size, replace=False)

        # gather sequences of transitions for each sampled start index
        batch_observations = np.array(
            [gather_sequence(self.observations, idx, sequence_length) for idx in start_indices]
        )
        batch_actions = np.array([gather_sequence(self.actions, idx, sequence_length) for idx in start_indices])
        batch_rewards = np.array([gather_sequence(self.rewards, idx, sequence_length) for idx in start_indices])
        batch_dones = np.array([gather_sequence(self.dones, idx, sequence_length) for idx in start_indices])
        return {
            "observations": batch_observations,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "dones": batch_dones,
        }


def gather_sequence(array, start_index, length):
    """Gather a sequence of elements from the input array starting from the specified index and wrapping around if necessary."""
    end_index = start_index + length
    if end_index <= len(array):
        # no wrap-around needed, simply slice the array
        return array[start_index:end_index]
    else:
        # concatenate the end of the array with the beginning to achieve wrap-around
        return np.concatenate((array[start_index:], array[: end_index % len(array)]))
