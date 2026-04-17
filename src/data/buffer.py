import numpy as np
import torch


class ReplayBuffer:
    """Replay buffer utility class.

    Responsible for storing experience data, and providing batch sampling.
    """

    def __init__(
        self,
        observation_shape: tuple,
        action_size: int,
        capacity: int,
        dtype: str = "uint8",
    ):
        """Construct an empty replay buffer.

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): action size.
            capacity (int): maximum capacity.
            dtype (str): buffer dtype.
        """
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be greater than zero.")

        self.capacity = capacity
        self.is_full = False
        self.buffer_index = 0  # pointer to the next index to insert a transition

        # internal buffer arrays to store transitions
        self.observations = np.zeros((self.capacity, *observation_shape), dtype=dtype)
        self.actions = np.zeros((self.capacity, action_size), dtype=dtype)
        self.rewards = np.zeros((self.capacity,), dtype=dtype)
        self.dones = np.zeros((self.capacity,), dtype=bool)

    def __len__(self) -> int:
        """Get the current number of transitions stored in the buffer."""
        return self.capacity if self.is_full else self.buffer_index

    def add(self, observation, action, reward, done):
        """Add a transition to the replay buffer

        The next observation is not stored, because this is a sequential replay
        buffer, and it can be easily fetched in a batch by incrementing the
        index by one.

        Old transitions are overwritten first if buffer capacity is exceeded.

        Args:
            observation: the current observation.
            action: the action taken.
            reward: the next observed reward.
            done: whether the episode is done.
        """

        # convert to numpy arrays if tensors are givens
        if torch.is_tensor(observation):
            observation = observation.numpy(force=True)
        if torch.is_tensor(action):
            action = action.numpy(force=True)

        self.observations[self.buffer_index] = observation
        self.actions[self.buffer_index] = action
        self.rewards[self.buffer_index] = reward
        self.dones[self.buffer_index] = done

        # increment buffer index and wrap around if we exceed capacity, overwriting old transitions
        self.buffer_index = (self.buffer_index + 1) % self.capacity
        if self.buffer_index == 0:
            self.is_full = True

    def sample(self, batch_size, sequence_length) -> dict:
        """Sample a batch of sequences of transitions from the replay buffer.

        A single sequence may contain multiple episode trajectories if the
        episode lengths are short enough. It is the caller's responsibility
        to reset recurrent state or memory according to the done flag.

        Args:
            batch_size (int): number of sequences in batch.
            sequence_length (int): length of each sequence in batch.

        Returns:
            batch (dict): dictionary containing batch observations, actions, rewards, and dones.

        """
        if len(self) < sequence_length:
            raise ValueError("Buffer not full enough.")

        # curate the pool of available start indices to sample
        if not self.is_full:
            # if not full yet, only sample from sequences that meet desired length.
            # anything that crosses beyond write boundary is invalid data
            max_start_index = self.buffer_index - sequence_length
            start_index_pool = np.arange(0, max_start_index + 1)
        else:
            if self.dones[(self.buffer_index - 1) % self.capacity]:
                # the last written timestep marks a valid episode boundary,
                # we can use all indices in buffer
                start_index_pool = np.arange(0, self.capacity)
            else:
                # last written timestep is NOT a valid episode boundary,
                # cannot sample sequences that cross buffer write boundary
                # without introducing silent discontinuity which would appear
                # as a random "jump" between two episodes, viewed as one.
                start_index_pool = np.concatenate(
                    [
                        # buffer index onwards to end of buffer
                        np.arange(self.buffer_index, self.capacity),
                        # beginning of buffer to first valid index before write boundary
                        np.arange(0, self.buffer_index - sequence_length + 1),
                    ]
                )

        if len(start_index_pool) < batch_size:
            raise ValueError("Batch size could not be fulfilled.")

        start_indices = np.random.choice(start_index_pool, size=batch_size, replace=False)
        indices = (start_indices[:, np.newaxis] + np.arange(sequence_length)) % self.capacity

        batch_observations = self.observations[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = self.dones[indices]

        return {
            "observations": batch_observations,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "dones": batch_dones,
        }
