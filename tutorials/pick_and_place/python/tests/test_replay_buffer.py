"""Unit tests for replay_buffer.ReplayBuffer."""

import numpy as np
import numpy.typing as npt
import pytest
from replay_buffer import ReplayBuffer


def make_batch(ids: range) -> tuple[npt.NDArray[np.float32], ...]:
    """A batch of len(ids) transitions where state == id, for easy tracking of
    which transitions survive in the buffer; every other field is zeroed out."""
    n = len(ids)
    states = np.array(ids, dtype=np.float32).reshape(n, 1)
    zeros = np.zeros((n, 1), dtype=np.float32)
    return states, zeros, zeros, zeros, zeros


def test_sample_from_empty_buffer_raises():
    buf = ReplayBuffer(capacity=10, state_dim=3, action_dim=2)
    assert len(buf) == 0
    with pytest.raises(ValueError):
        buf.sample(2)


def test_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=10, state_dim=3, action_dim=2)
    states = np.arange(5 * 3).reshape(5, 3).astype(np.float32)
    actions = np.arange(5 * 2).reshape(5, 2).astype(np.float32)
    rewards = np.arange(5).reshape(5, 1).astype(np.float32)
    next_states = states + 100
    terminations = np.zeros((5, 1), dtype=np.float32)

    buf.add(states, actions, rewards, next_states, terminations)
    assert len(buf) == 5

    batch = buf.sample(4)
    assert batch.states.shape == (4, 3)
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4, 1)
    assert batch.next_states.shape == (4, 3)
    assert batch.terminations.shape == (4, 1)


def test_fifo_eviction_across_multiple_adds():
    """capacity=10, adding 5+5+5=15 transitions should evict the oldest 5,
    leaving exactly the most recent 10."""
    buf = ReplayBuffer(capacity=10, state_dim=1, action_dim=1)
    buf.add(*make_batch(range(0, 5)))
    buf.add(*make_batch(range(5, 10)))
    buf.add(*make_batch(range(10, 15)))

    assert len(buf) == 10
    remaining_ids = sorted(buf.states.flatten().tolist())
    assert remaining_ids == list(range(5, 15))


def test_oversized_single_batch_truncates_to_capacity():
    """A single add() bigger than the whole buffer should keep only the most
    recent `capacity` rows, not silently overflow or wrap incorrectly."""
    buf = ReplayBuffer(capacity=10, state_dim=1, action_dim=1)
    buf.add(*make_batch(range(0, 25)))

    assert len(buf) == 10
    remaining_ids = sorted(buf.states.flatten().tolist())
    assert remaining_ids == list(range(15, 25))


def test_partial_fill_never_samples_stale_zero_init_rows():
    """Before the buffer has filled to capacity, sample() must only ever
    return written-to rows, never the zero-initialized backing array."""
    buf = ReplayBuffer(capacity=100, state_dim=1, action_dim=1)
    buf.add(*make_batch(range(1, 4)))  # avoid 0 so a leaked zero-init row is detectable

    for _ in range(50):
        batch = buf.sample(5)
        assert set(batch.states.flatten().tolist()) <= {1.0, 2.0, 3.0}
