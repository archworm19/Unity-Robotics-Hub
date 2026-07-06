"""Common inference interface for anything that maps the current sim state to
the next action -- a trained model (see train_bc_mlp.BehaviorCloningMLP), and
Replay, which plays back a previously recorded trajectory for sanity-checking.

Interface: act(state) -> action, both flat arrays matching
remote_connection.observation_to_array / remote_connection.send_action's
layout. Deliberately just a function of the current state -- anything that
needs history (an RNN policy, a smoothing filter, ...) is expected to keep
that internally rather than have it threaded through the caller.

Construction goes through load_from_checkpoint(data_location,
checkpoint_location), the same two-argument signature for every
implementation so a generic caller doesn't need to know which kind of policy
it's loading. Each implementation only actually uses whichever of the two
locations is relevant to it and ignores the other -- Replay only needs
data_location (the recorded actions to play back, it has no trained
checkpoint), while a trained model only needs checkpoint_location (its saved
weights self-contain everything else needed). BehaviorCloningMLP satisfies
this Policy interface structurally (see train_bc_mlp.py) rather than being
imported and wrapped here, so this module doesn't need to depend on torch.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Policy(Protocol):
    def act(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Given the current state, return the next action to send."""
        ...

    @classmethod
    def load_from_checkpoint(cls, data_location: Path, checkpoint_location: Path) -> "Policy":
        """Construct a Policy from wherever its data/checkpoint artifacts live."""
        ...


class Replay:
    """Plays back a fixed, pre-recorded sequence of actions, ignoring whatever
    state is actually passed to act() -- for sanity-checking that a recorded
    trajectory (see teleop_controller.py) reproduces a sensible run when
    resent into the sim, not an actual policy.

    Recording only keeps ticks where the action changed (see
    teleop_controller.RECORD_ONLY_NONZERO_ACTIONS), so resending each recorded
    action exactly once would compress out the real "holding position" ticks
    that were skipped, and PD-controlled joints wouldn't have time to actually
    converge on each target before the next one arrived. pause_timesteps
    repeats each action that many extra times before advancing to the next
    one, to approximate a human having actually held it for a few ticks.
    """

    def __init__(
        self,
        actions: npt.NDArray[np.float64],
        pause_timesteps: int = 2,
        states: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self._actions = actions
        self._pause_timesteps = pause_timesteps
        self._states = states  # ground-truth recorded states, for drift() -- optional
        self._iterator = self._make_iterator()
        self._finished = False
        self._current_index: int | None = None
        self._next_action, self._next_index = self._advance()

    @classmethod
    def load_from_checkpoint(cls, data_location: Path, checkpoint_location: Path) -> "Replay":
        """data_location is the recorded actions.npy to play back (Replay has no
        trained weights, so checkpoint_location is unused). If a states_*.npy
        exists alongside it under the matching name (see teleop_controller.py's
        save_history), it's loaded too, enabling drift()."""
        del checkpoint_location
        actions = np.load(data_location)
        states_path = data_location.parent / data_location.name.replace("actions_", "states_", 1)
        states = np.load(states_path) if states_path != data_location and states_path.exists() else None
        return cls(actions, states=states)

    def _make_iterator(self):
        for index, action in enumerate(self._actions):
            yield action, index
            for _ in range(self._pause_timesteps):
                yield action, index

    def _advance(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._finished = True
            return None, None

    @property
    def finished(self) -> bool:
        """True once every recorded action (and its pause repeats) has been returned."""
        return self._finished

    @property
    def current_index(self) -> int | None:
        """Index into the original recorded actions array that the most recently
        returned action (from act()) came from. None before the first act() call."""
        return self._current_index

    def drift(self, live_state: npt.NDArray[np.float64]) -> float | None:
        """Distance between live_state's end-effector position and what was
        originally recorded at the current replay index -- None if no
        ground-truth states were loaded (see load_from_checkpoint), or before
        the first act() call."""
        if self._states is None or self._current_index is None:
            return None
        recorded_eef = self._states[self._current_index][0:3]
        live_eef = live_state[0:3]
        return float(np.linalg.norm(live_eef - recorded_eef))

    def act(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        del state  # Replay ignores the observed state; it just plays back a fixed script.
        if self._finished:
            raise IndexError("Replay.act called after the recorded trajectory finished.")
        action, index = self._next_action, self._next_index
        self._current_index = index
        self._next_action, self._next_index = self._advance()
        return action
