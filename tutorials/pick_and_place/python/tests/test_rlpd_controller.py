"""Unit tests for rlpd_controller's reward computation, its four pending-
transition relabeling behaviors (intervention start/end, and IK solve
failures) -- specifically that each is synced correctly to the tick it's
meant to fire on, never double-fires, and never clobbers a real terminal
reward -- and its delta/absolute arm-action conversion."""

import numpy as np
import pytest
import rlpd_controller as rcm
from remote_connection import Observation, PlacementState

STATE = np.zeros(4, dtype=np.float32)
ACTION = np.zeros(2, dtype=np.float32)

# bounds=(-1, 1) makes ik.angles_to_normalized an identity map, so
# "current normalized position" in the tests below is just whatever
# joint_positions_radians already is -- no need to load a real IK chain.
SIMPLE_BOUNDS = [(-1.0, 1.0)] * 6


def make_pending(reward: float = 0.0, done: bool = False) -> rcm.PendingTransition:
    return rcm.PendingTransition(
        pre_state=STATE, action=ACTION, reward=reward, post_state=STATE, done=done, override_action=None
    )


def make_observation(joint_positions_radians) -> Observation:
    return Observation(
        end_effector_position_ruf=np.zeros(3),
        end_effector_orientation_ruf=np.array([0.0, 0.0, 0.0, 1.0]),
        gripper_width_meters=0.02,
        joint_positions_radians=np.array(joint_positions_radians, dtype=np.float64),
        goal_position_ruf=np.zeros(3),
        block_position_ruf=np.zeros(3),
        block_orientation_ruf=np.array([0.0, 0.0, 0.0, 1.0]),
        placement_state=PlacementState.OUTSIDE,
    )


def test_compute_reward_success_is_terminal():
    reward, done = rcm.compute_reward(PlacementState.INSIDE_PLACED, block_height_above_table=0.3)
    assert reward == rcm.SUCCESS_REWARD
    assert done is True


def test_compute_reward_failure_is_terminal():
    reward, done = rcm.compute_reward(PlacementState.FAILED_FELL, block_height_above_table=0.0)
    assert reward == rcm.FAILURE_REWARD
    assert done is True


def test_compute_reward_terminal_states_ignore_block_height():
    """Success/failure are read straight off placement_state -- block height
    shouldn't matter at all once the episode is already terminal."""
    reward, _ = rcm.compute_reward(PlacementState.INSIDE_PLACED, block_height_above_table=999.0)
    assert reward == rcm.SUCCESS_REWARD


def test_compute_reward_resting_on_table():
    for state in (PlacementState.OUTSIDE, PlacementState.INSIDE_FLOATING):
        reward, done = rcm.compute_reward(state, block_height_above_table=0.0)
        assert reward == rcm.RESTING_ON_TABLE_REWARD
        assert done is False


def test_compute_reward_in_air():
    for state in (PlacementState.OUTSIDE, PlacementState.INSIDE_FLOATING):
        reward, done = rcm.compute_reward(state, block_height_above_table=0.5)
        assert reward == rcm.IN_AIR_REWARD
        assert done is False


def test_compute_reward_resting_tolerance_boundary():
    at_tolerance, _ = rcm.compute_reward(PlacementState.OUTSIDE, rcm.TABLE_REST_TOLERANCE_METERS)
    just_above, _ = rcm.compute_reward(PlacementState.OUTSIDE, rcm.TABLE_REST_TOLERANCE_METERS + 1e-6)
    assert at_tolerance == rcm.RESTING_ON_TABLE_REWARD
    assert just_above == rcm.IN_AIR_REWARD


def test_validate_reward_magnitudes_raises_when_handback_exceeds_intervention_penalty():
    with pytest.raises(ValueError):
        rcm.validate_reward_magnitudes(handback_reward=2.0, intervention_penalty=-1.0)


def test_validate_reward_magnitudes_allows_equal_magnitude():
    rcm.validate_reward_magnitudes(handback_reward=1.0, intervention_penalty=-1.0)  # should not raise


def test_validate_reward_magnitudes_allows_smaller_handback():
    rcm.validate_reward_magnitudes(handback_reward=0.5, intervention_penalty=-2.0)  # should not raise


def test_relabel_when_intervention_ends():
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_ended(pending, intervening=False, was_intervening=True)
    assert result.reward == rcm.HANDBACK_REWARD


def test_no_relabel_when_intervention_has_not_ended():
    pending = make_pending(reward=0.0, done=False)
    # still intervening -- no end-of-intervention transition this tick
    result = rcm.relabel_pending_if_intervention_ended(pending, intervening=True, was_intervening=True)
    assert result.reward == 0.0
    # was never intervening in the first place
    result = rcm.relabel_pending_if_intervention_ended(pending, intervening=False, was_intervening=False)
    assert result.reward == 0.0


def test_relabel_intervention_ended_does_not_clobber_already_terminal_pending():
    pending = make_pending(reward=rcm.FAILURE_REWARD, done=True)
    result = rcm.relabel_pending_if_intervention_ended(pending, intervening=False, was_intervening=True)
    assert result.reward == rcm.FAILURE_REWARD
    assert result.done is True


def test_relabel_intervention_ended_disabled_by_toggle(monkeypatch):
    monkeypatch.setattr(rcm, "REWARD_ACTION_BEFORE_HANDBACK", False)
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_ended(pending, intervening=False, was_intervening=True)
    assert result.reward == 0.0


def test_intervention_start_and_end_relabels_are_mutually_exclusive():
    """Chaining both relabel calls (as main() does) must never let one
    transition's relabel be immediately overwritten by the other."""
    pending = make_pending(reward=0.0, done=False)
    started = rcm.relabel_pending_if_intervention_started(pending, intervening=True, was_intervening=False)
    both = rcm.relabel_pending_if_intervention_ended(started, intervening=True, was_intervening=False)
    assert both.reward == rcm.INTERVENTION_PENALTY  # the "ended" call was a no-op, since intervening=True here


def test_relabel_when_intervention_starts():
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_started(pending, intervening=True, was_intervening=False)
    assert result.reward == rcm.INTERVENTION_PENALTY


def test_relabel_preserves_other_fields():
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_started(pending, intervening=True, was_intervening=False)
    assert result is not pending  # dataclasses.replace returns a new instance
    assert np.array_equal(result.pre_state, pending.pre_state)
    assert np.array_equal(result.action, pending.action)
    assert np.array_equal(result.post_state, pending.post_state)
    assert result.done == pending.done
    assert result.override_action == pending.override_action


def test_no_relabel_when_intervention_continues():
    """Only the single tick where intervening flips False -> True should relabel --
    every subsequent tick of the same, ongoing intervention must not re-trigger it."""
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_started(pending, intervening=True, was_intervening=True)
    assert result.reward == 0.0


def test_no_relabel_when_not_intervening():
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_started(pending, intervening=False, was_intervening=False)
    assert result.reward == 0.0


def test_no_relabel_when_pending_is_none():
    result = rcm.relabel_pending_if_intervention_started(None, intervening=True, was_intervening=False)
    assert result is None


def test_no_relabel_when_pending_already_terminal():
    """A human intervening right after a fresh (post-reset) episode begins must
    not overwrite the previous, genuinely terminal success/failure reward."""
    pending = make_pending(reward=rcm.SUCCESS_REWARD, done=True)
    result = rcm.relabel_pending_if_intervention_started(pending, intervening=True, was_intervening=False)
    assert result.reward == rcm.SUCCESS_REWARD
    assert result.done is True


def test_relabel_disabled_by_toggle(monkeypatch):
    monkeypatch.setattr(rcm, "PENALIZE_ACTION_BEFORE_INTERVENTION", False)
    pending = make_pending(reward=0.0, done=False)
    result = rcm.relabel_pending_if_intervention_started(pending, intervening=True, was_intervening=False)
    assert result.reward == 0.0


def test_ik_failure_marks_pending_terminal():
    pending = make_pending(reward=0.0, done=False)
    result = rcm.mark_pending_as_terminal_failure(pending, rcm.IK_FAILURE_PENALTY)
    assert result.reward == rcm.IK_FAILURE_PENALTY
    assert result.done is True


def test_ik_failure_preserves_other_fields():
    pending = make_pending(reward=0.0, done=False)
    result = rcm.mark_pending_as_terminal_failure(pending, rcm.IK_FAILURE_PENALTY)
    assert result is not pending
    assert np.array_equal(result.pre_state, pending.pre_state)
    assert np.array_equal(result.action, pending.action)
    assert np.array_equal(result.post_state, pending.post_state)
    assert result.override_action == pending.override_action


def test_ik_failure_does_not_clobber_already_terminal_pending():
    """An IK failure right after a fresh (post-reset) episode begins must not
    overwrite the previous, genuinely terminal success/failure reward."""
    pending = make_pending(reward=rcm.SUCCESS_REWARD, done=True)
    result = rcm.mark_pending_as_terminal_failure(pending, rcm.IK_FAILURE_PENALTY)
    assert result.reward == rcm.SUCCESS_REWARD
    assert result.done is True


def test_ik_failure_with_no_pending_returns_none():
    result = rcm.mark_pending_as_terminal_failure(None, rcm.IK_FAILURE_PENALTY)
    assert result is None


def test_episode_return_delta_is_zero_when_reward_unchanged():
    pending = make_pending(reward=0.0, done=False)
    assert rcm.episode_return_delta_for_relabel(0.0, pending) == 0.0


def test_episode_return_delta_reflects_relabel():
    """episode_return already added the pre-relabel reward a tick ago -- the
    delta must be new minus old, not the new reward on its own, or a relabel
    would double-count (or drop) whatever was already added."""
    relabeled = make_pending(reward=rcm.INTERVENTION_PENALTY, done=False)
    assert rcm.episode_return_delta_for_relabel(0.0, relabeled) == rcm.INTERVENTION_PENALTY - 0.0


def test_episode_return_delta_is_zero_when_pending_is_none():
    assert rcm.episode_return_delta_for_relabel(None, None) == 0.0


def test_delta_action_to_absolute_zero_delta_holds_position():
    observation = make_observation([0.2, -0.3, 0.1, 0.0, 0.4, -0.1])
    delta_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])  # 7th slot is gripper
    absolute = rcm.delta_action_to_absolute(delta_action, observation, SIMPLE_BOUNDS)
    assert np.allclose(absolute[:6], observation.joint_positions_radians)


def test_delta_action_to_absolute_scales_by_max_delta_per_tick():
    observation = make_observation(np.zeros(6))
    delta_action = np.concatenate([np.ones(6), [0.0]])
    absolute = rcm.delta_action_to_absolute(delta_action, observation, SIMPLE_BOUNDS)
    assert np.allclose(absolute[:6], rcm.ARM_MAX_DELTA_PER_TICK)


def test_delta_action_to_absolute_clips_to_valid_range():
    observation = make_observation(np.full(6, 0.999))
    delta_action = np.concatenate([np.ones(6), [0.0]])  # would overshoot 1.0 without clipping
    absolute = rcm.delta_action_to_absolute(delta_action, observation, SIMPLE_BOUNDS)
    assert np.all(absolute[:6] <= 1.0)


def test_delta_action_to_absolute_passes_gripper_through_unchanged():
    observation = make_observation(np.zeros(6))
    delta_action = np.concatenate([np.zeros(6), [-0.42]])
    absolute = rcm.delta_action_to_absolute(delta_action, observation, SIMPLE_BOUNDS)
    assert absolute[6] == -0.42


def test_absolute_action_to_delta_is_inverse_of_delta_action_to_absolute():
    observation = make_observation([0.1, -0.2, 0.3, -0.1, 0.05, 0.0])
    delta_action = np.concatenate([np.full(6, 0.3), [0.5]])  # small enough not to clip on the way back
    absolute = rcm.delta_action_to_absolute(delta_action, observation, SIMPLE_BOUNDS)
    recovered = rcm.absolute_action_to_delta(absolute, observation, SIMPLE_BOUNDS)
    assert np.allclose(recovered[:6], delta_action[:6], atol=1e-6)


def test_absolute_action_to_delta_clips_large_jumps():
    """A human's IK-solved override can imply a much bigger jump than the
    policy's own step size -- what's stored must still be a valid delta-space
    action ([-1, 1]), even though what's sent to Unity isn't rate-limited."""
    observation = make_observation(np.zeros(6))
    absolute_action = np.concatenate([np.ones(6), [0.0]])  # a full-range jump
    delta = rcm.absolute_action_to_delta(absolute_action, observation, SIMPLE_BOUNDS)
    assert np.allclose(delta[:6], 1.0)


def test_absolute_action_to_delta_passes_gripper_through_unchanged():
    observation = make_observation(np.zeros(6))
    absolute_action = np.concatenate([np.zeros(6), [0.9]])
    delta = rcm.absolute_action_to_delta(absolute_action, observation, SIMPLE_BOUNDS)
    assert delta[6] == 0.9


def test_full_tick_sequence_relabels_exactly_the_expected_transitions():
    """End-to-end replay of a tick-by-tick (intervening, reward, done) sequence
    through the same chained relabel-then-flush pattern main() uses (both
    start-of-intervention and end-of-intervention relabels), mirroring a
    realistic session: autonomous driving, a human takeover mid-episode, a
    deliberate handback, a genuine success, and a human intervening again
    right after the reset."""
    ticks = [
        (False, 0.0, False),  # 0: autonomous
        (False, 0.0, False),  # 1: autonomous -- last action before takeover, should be penalized
        (True, 0.0, False),  # 2: human takes over
        (True, 0.0, False),  # 3: still human -- last human action before handback, should be rewarded
        (False, 0.0, False),  # 4: back to autonomous
        (False, 1.0, True),  # 5: success (terminal) -- must survive untouched
        (True, 0.0, False),  # 6: human intervenes right after the reset
    ]

    stored: list[tuple[float, bool]] = []
    pending: rcm.PendingTransition | None = None
    was_intervening = False

    for intervening, reward, done in ticks:
        pending = rcm.relabel_pending_if_intervention_started(pending, intervening, was_intervening)
        pending = rcm.relabel_pending_if_intervention_ended(pending, intervening, was_intervening)
        if pending is not None:
            stored.append((pending.reward, pending.done))
        pending = make_pending(reward=reward, done=done)
        was_intervening = intervening
    if pending is not None:
        stored.append((pending.reward, pending.done))

    assert stored == [
        (0.0, False),
        (rcm.INTERVENTION_PENALTY, False),
        (0.0, False),
        (rcm.HANDBACK_REWARD, False),
        (0.0, False),
        (1.0, True),
        (0.0, False),
    ]
