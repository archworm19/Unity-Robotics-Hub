"""Fixed-capacity FIFO replay buffer for off-policy RL (DQN/DDPG/TD3/SAC-style
algorithms): add batches of transitions, sample uniformly at random for training.

Standard design: pre-allocated arrays sized (capacity, dim) per field, a write
pointer that wraps around once full (oldest transitions get overwritten --
FIFO eviction, O(1) per insertion, no shifting), and uniform-random-with-
replacement sampling over whatever's currently stored. Single-threaded for
now -- no locking, since only one thread adds/samples at a time.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Batch:
    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    rewards: npt.NDArray[np.float32]
    next_states: npt.NDArray[np.float32]
    terminations: npt.NDArray[np.float32]


class ReplayBuffer:
    """add() and sample() both work in batches: every field is shaped
    (num_samples, d), including rewards/terminations (d=1) -- callers are
    expected to pass them that way rather than as bare 1-D arrays.

    next_states is stored alongside states/actions/rewards/terminations so a
    TD bootstrap target can be computed later without needing to reconstruct
    it (e.g. by shifting states by one row, which breaks at episode
    boundaries) -- every standard off-policy algorithm needs it.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminations = np.zeros((capacity, 1), dtype=np.float32)
        self._write_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        states: npt.NDArray[np.float32],
        actions: npt.NDArray[np.float32],
        rewards: npt.NDArray[np.float32],
        next_states: npt.NDArray[np.float32],
        terminations: npt.NDArray[np.float32],
    ) -> None:
        """Each argument is shaped (num_samples, d) -- call with num_samples=1 for
        a single transition, or more to add a whole batch/episode at once."""
        num_samples = states.shape[0]

        if num_samples > self.capacity:
            # Keep only the most recent `capacity` rows -- anything older would
            # just get immediately overwritten by the wraparound below anyway.
            start = num_samples - self.capacity
            states, actions, rewards = states[start:], actions[start:], rewards[start:]
            next_states, terminations = next_states[start:], terminations[start:]
            num_samples = self.capacity

        indices = (self._write_index + np.arange(num_samples)) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.terminations[indices] = terminations

        self._write_index = (self._write_index + num_samples) % self.capacity
        self._size = min(self.capacity, self._size + num_samples)

    def sample(self, batch_size: int) -> Batch:
        """Uniform random sample, with replacement, from whatever's currently
        stored (only ever the valid, written-to portion, even before the
        buffer has filled up to capacity for the first time)."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")
        indices = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            terminations=self.terminations[indices],
        )
