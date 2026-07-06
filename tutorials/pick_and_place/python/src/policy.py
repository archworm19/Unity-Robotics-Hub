"""Common inference interface for anything that maps the current sim state to
the next action -- a trained model eventually, and for now Replay, which plays
back a previously recorded trajectory for sanity-checking.

Interface: act(state) -> action, both flat arrays matching
remote_connection.observation_to_array / remote_connection.send_action's
layout. Deliberately just a function of the current state -- anything that
needs history (an RNN policy, a smoothing filter, ...) is expected to keep
that internally rather than have it threaded through the caller.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Policy(Protocol):
    def act(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Given the current state, return the next action to send."""
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

    def __init__(self, actions: npt.NDArray[np.float64], pause_timesteps: int = 2) -> None:
        self._actions = actions
        self._pause_timesteps = pause_timesteps
        self._iterator = self._make_iterator()
        self._finished = False
        self._current_index: int | None = None
        self._next_action, self._next_index = self._advance()

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

    def act(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        del state  # Replay ignores the observed state; it just plays back a fixed script.
        if self._finished:
            raise IndexError("Replay.act called after the recorded trajectory finished.")
        action, index = self._next_action, self._next_index
        self._current_index = index
        self._next_action, self._next_index = self._advance()
        return action
