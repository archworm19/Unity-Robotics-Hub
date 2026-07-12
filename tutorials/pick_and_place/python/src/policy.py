"""Common inference interface for anything that maps the current sim state to
the next action -- a trained model (see train_bc_mlp.BehaviorCloningMLP), and
Replay, which plays back a previously recorded trajectory for sanity-checking.
Also HILPolicy, the variant used during human-in-the-loop RL training.

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


class HILPolicy(Policy, Protocol):
    """Everything Policy requires (act, load_from_checkpoint, unchanged), plus
    store_transition for human-in-the-loop RL training. act() never takes a
    hil_action -- it always just proposes the policy's own action; the
    training loop is what decides, from a separate human-intervention signal,
    whether to actually execute that or something else instead, and reports
    the outcome back via store_transition.

    A HILPolicy owns its own replay buffer and update schedule entirely
    internally -- store_transition is the only thing the training loop calls
    per step; there's no separate update()/learn() method in this interface,
    since e.g. a Q-learning policy would just run its gradient step at the
    end of store_transition() once it has enough data.
    """

    def store_transition(
        self,
        pre_state: npt.NDArray[np.float64],
        action: npt.NDArray[np.float64],
        reward: float,
        post_state: npt.NDArray[np.float64],
        done: bool,
        override_action: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Record one complete (pre_state, action, reward, post_state, done)
        transition for learning, and update if/however this policy decides to.

        action is whatever this policy's own act(pre_state) proposed.
        override_action is the action that was actually executed instead, if a
        human intervened this step (None otherwise) -- the policy decides how
        to use both: e.g. storing override_action (what really happened) as
        the transition's action in its own replay buffer rather than action
        (what it would have done), since that's what it should learn from.
        reward is expected to already reflect any caller-side adjustments
        (e.g. an intervention penalty when override_action is set) -- this
        interface just records it, it doesn't compute or adjust it.
        """
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
        self._last_action: npt.NDArray[np.float64] | None = None
        self._current_index: int | None = None
        self._calls_since_last_inference = 0

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

    @property
    def finished(self) -> bool:
        """True once every recorded action (and its pause repeats) has been returned."""
        if self._current_index is None:
            return len(self._actions) == 0
        return (
            self._current_index == len(self._actions) - 1
            and self._calls_since_last_inference >= self._pause_timesteps
        )

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
        if self.finished:
            raise IndexError("Replay.act called after the recorded trajectory finished.")

        if self._current_index is None:
            self._advance_to_next_action()
        else:
            self._calls_since_last_inference += 1
            if self._calls_since_last_inference > self._pause_timesteps:
                self._advance_to_next_action()

        return self._last_action

    def _advance_to_next_action(self) -> None:
        self._current_index = 0 if self._current_index is None else self._current_index + 1
        self._last_action = self._actions[self._current_index]
        self._calls_since_last_inference = 0
