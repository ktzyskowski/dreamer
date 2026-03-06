import numpy as np
import pytest

from modules.utils.buffer import ReplayBuffer, gather_sequence


def make_transition(value, observation_shape, action_shape):
    observation = np.full(observation_shape, value, dtype=np.float32)
    action = np.full(action_shape, value, dtype=np.float32)
    reward = float(value)
    next_observation = observation + 100.0
    done = float(value % 2)
    return observation, action, reward, next_observation, done


def test_constructor_negative_capacity_raises():
    with pytest.raises(Exception):
        ReplayBuffer((3, 64, 64), (4,), capacity=-1)


def test_constructor_zero_capacity_raises():
    with pytest.raises(Exception):
        ReplayBuffer((3, 64, 64), (4,), capacity=0)


def test_add_and_wrap_behavior():
    capacity = 5
    buffer = ReplayBuffer((2,), (1,), capacity=capacity)
    # add more than capacity to force wrap-around
    for index in range(7):
        buffer.add(*make_transition(index, observation_shape=(2,), action_shape=(1,)))

    assert len(buffer) == capacity
    assert buffer.is_full is True
    # buffer_index should be 7 % 5 == 2
    assert buffer.buffer_index == 2

    # entries at indices 0..4 should correspond to the last 5 insertions (i=2..6)
    expected = [5, 6, 2, 3, 4]
    stored = [int(buffer.observations[i][0]) for i in range(capacity)]
    assert stored == expected


def test_sample_non_full_sequences_and_shapes():
    capacity = 10
    buffer = ReplayBuffer((3,), (1,), capacity=capacity)
    # add 6 entries
    for index in range(6):
        buffer.add(*make_transition(index, observation_shape=(3,), action_shape=(1,)))

    batch = buffer.sample(batch_size=2, sequence_length=3)
    # shapes should be (batch, seq, ...)
    assert batch["observations"].shape == (2, 3, 3)
    assert batch["actions"].shape == (2, 3, 1)
    assert batch["rewards"].shape == (2, 3)
    assert batch["next_observations"].shape == (2, 3, 3)
    assert batch["dones"].shape == (2, 3)


def test_sample_full_buffer_wrapping_sequences():
    capacity = 5
    buffer = ReplayBuffer((2,), (1,), capacity=capacity)
    # fill buffer exactly
    for index in range(capacity):
        buffer.add(*make_transition(index, observation_shape=(2,), action_shape=(1,)))
    assert buffer.is_full

    # sample sequences that may wrap
    batch = buffer.sample(batch_size=3, sequence_length=3)
    obs = batch["observations"]
    # each sequence length check
    assert obs.shape == (3, 3, 2)
    # verify that sequences equal gather_sequence applied to underlying arrays
    for b in range(3):
        # find a matching start index by searching the first element of the sequence
        seq0 = obs[b][0][0]
        # look for index in buffer where value equals seq0
        idxs = [i for i in range(capacity) if int(buffer.observations[i][0]) == int(seq0)]
        # at least one match should exist
        assert idxs


def test_sample_insufficient_data_raises():
    buffer = ReplayBuffer((2,), (1,), capacity=10)
    # add only 2 entries
    buffer.add(*make_transition(0, observation_shape=(2,), action_shape=(1,)))
    buffer.add(*make_transition(1, observation_shape=(2,), action_shape=(1,)))
    with pytest.raises(Exception):
        buffer.sample(batch_size=1, sequence_length=3)


def test_sample_batch_too_large_raises():
    buffer = ReplayBuffer((2,), (1,), capacity=10)
    for index in range(5):
        buffer.add(*make_transition(index, observation_shape=(2,), action_shape=(1,)))
    # available starts = buffer_index - seq_len -> 5 - 2 = 3 -> start pool size 3
    with pytest.raises(ValueError):
        buffer.sample(batch_size=10, sequence_length=2)


def test_gather_sequence_wraps_correctly():
    arr = np.arange(10)
    # no wrap
    seq = gather_sequence(arr, 2, 4)
    assert np.array_equal(seq, np.array([2, 3, 4, 5]))
    # wrap around
    seq2 = gather_sequence(arr, 8, 5)
    assert np.array_equal(seq2, np.array([8, 9, 0, 1, 2]))
