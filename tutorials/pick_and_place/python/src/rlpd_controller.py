"""Human-in-the-loop RLPD training against the live Unity task (see policy.py's
HILPolicy and rlpd_policy.RLPDPolicy). Distinct from policy_controller.py,
which only ever runs inference for a fixed Policy -- this loop trains as it
goes: every tick it proposes an action, optionally lets a human override it via
the same pygame/spherical-coordinate scheme as teleop_controller.py, computes a
reward from the resulting placement_state, and calls store_transition().

Episode boundaries are entirely Unity-driven: RemoteJointController resets the
block automatically once placement_state comes back INSIDE_PLACED (success) or
FAILED_FELL (fell off the table) -- see remote_connection.py's module
docstring -- so this loop never sends anything resembling a reset command; it
just keeps streaming actions and reacts to whatever placement_state comes
back. There's currently no way for a human to abort a stuck episode early
(e.g. an arm pose that's technically fine but going nowhere) short of
physically knocking the block off the table -- only Unity's own success/fall
detection ends an episode.

Reward: +1 on success, -1 on falling off the table (both terminal); otherwise
a small dense shaping term from the block's height relative to its per-episode
table/spawn height -- RESTING_ON_TABLE_REWARD (small negative) while it's
sitting untouched, IN_AIR_REWARD (0) once it's been lifted at all. This is
meant as mild pressure to actually pick the block up rather than leave it
alone, not a proxy for the sparse success/failure signal.

PENALIZE_ACTION_BEFORE_INTERVENTION additionally relabels the reward of the
single policy action immediately preceding a human takeover to
INTERVENTION_PENALTY -- the (state, action) pair that made a human feel the
need to intervene is exactly the thing worth teaching the policy not to
repeat. Symmetrically, REWARD_ACTION_BEFORE_HANDBACK relabels the human's own
last action right before handing control back to the policy (see "t" toggle,
below) to HANDBACK_REWARD (small positive) -- the state a human chose to hand
back from is exactly the state worth teaching the policy to reach and
continue from on its own. HANDBACK_REWARD's magnitude must never exceed
INTERVENTION_PENALTY's -- a module-level check raises ValueError at import
time if that's violated, since a handback reward that could outweigh an
intervention penalty would undermine the whole point of the penalty. Both
require holding back each transition for one tick before actually calling
store_transition() on it, since whether to relabel it isn't known until the
*next* tick reveals a start/end-of-intervention transition. Only the single,
immediately-preceding transition is ever relabeled -- and never one that was
itself already terminal (done=True), since that would corrupt a real
success/failure signal with an unrelated intervention boundary that just
happened to land on the following, fresh episode.

Human intervention: press "t" to toggle teleop mode on/off -- while on, the
same WASD/Up/Down/Space controls as teleop_controller.py drive the arm
directly and the policy is completely ignored, even on ticks with no key held
(the arm just holds position); while off, the policy drives every tick and
WASD/Up/Down/Space do nothing. See InterventionController's docstring for why
this replaced an earlier momentary-key-hold design.

IK solve failures: seeding the IK solver from the arm's *live* joint angles
(above) means an untrained policy can drift the arm into a pose ikpy's own
solver considers out of bounds, which raises rather than returning a bad
result -- e.g. scipy's "Initial guess is outside of provided bounds". This
only ever surfaces while computing a human-intervention override (the policy
itself never calls into IK). Treated the same as any other unrecoverable
failure: the pending transition -- whichever action actually left the arm in
that pose -- is relabeled done=True with IK_FAILURE_PENALTY, and a reset is
requested that, unlike the automatic placement-driven one, resets the arm too
(see remote_connection.send_action's reset flag), since here the arm itself is
the thing that broke.

Delta rather than absolute arm control: the policy's own action space is a
bounded normalized *change* in position (ARM_MAX_DELTA_PER_TICK per tick) for
each of the 6 arm joints, not an absolute target -- an untrained/random policy
can then only ever nudge a joint from wherever it actually is, rather than
commanding wild jumps to arbitrary positions. The gripper command is
unaffected -- still an absolute open/closed target, as before. Unity's own
wire protocol never changed (it still expects absolute normalized targets, as
it always has); the delta/absolute conversion happens entirely here, in
Python: delta_action_to_absolute before sending an autonomous action, and its
inverse absolute_action_to_delta when storing a human's IK-solved
override_action, so the replay buffer's action semantics stay consistent
regardless of who actually executed a given transition.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).parent))
import ik
import remote_connection as rc
from remote_connection import Observation, PlacementState
from rlpd_policy import RLPDPolicy
from teleop_controller import (
    GRIPPER_CLOSED_COMMAND,
    GRIPPER_OPEN_COMMAND,
    GRIPPER_ROOT_LINK,
    GRIPPER_STEP_PER_TICK,
    NUM_POSITION_JOINTS,
    PHI_MAX_RADIANS,
    PHI_MIN_RADIANS,
    R_MAX_METERS,
    R_MIN_METERS,
    URDF_PATH,
    WRIST_JOINT_ANGLES,
    cartesian_to_spherical,
    compute_spherical_delta,
    spherical_to_cartesian,
    step_toward,
)

HOST = "127.0.0.1"
PORT = 9000
CONNECT_TIMEOUT_SECONDS = 30.0

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
DEFAULT_CHECKPOINT_PATH = TELEOP_DIRECTORY / "rlpd.pt"
CHECKPOINT_EVERY_EPISODES = 10

STATE_DIM = rc.NUM_OBSERVATION_FLOATS
ACTION_DIM = rc.NUM_POLICY_ACTION_FLOATS  # NUM_ACTION_FLOATS also counts the wire-only reset flag -- not a policy output

SUCCESS_REWARD = 5.0
FAILURE_REWARD = -5.0

# See module docstring's opening "Reward" paragraph -- dense shaping applied
# on every non-terminal tick, based on the block's height above its
# per-episode table/spawn reference (see main()'s table_height tracking).
TABLE_REST_TOLERANCE_METERS = 0.02  # how far above table height still counts as "resting", not lifted
RESTING_ON_TABLE_REWARD = 0.0
IN_AIR_REWARD = 0.0

# See module docstring: relabels the one policy action immediately preceding a
# human takeover with INTERVENTION_PENALTY instead of whatever compute_reward
# would otherwise have given it. Larger in magnitude than a plain failure --
# a human having to step in is a stronger negative signal than just letting
# the episode fail and reset on its own.
PENALIZE_ACTION_BEFORE_INTERVENTION = True
INTERVENTION_PENALTY = -25.0

# See module docstring: relabels the human's own last action right before
# handing control back to the policy with HANDBACK_REWARD instead of whatever
# compute_reward would otherwise have given it.
REWARD_ACTION_BEFORE_HANDBACK = True
HANDBACK_REWARD = 2.0

def validate_reward_magnitudes(handback_reward: float, intervention_penalty: float) -> None:
    """Raises ValueError if handback_reward's magnitude exceeds
    intervention_penalty's -- see REWARD_ACTION_BEFORE_HANDBACK in the module
    docstring for why that invariant must hold."""
    if abs(handback_reward) > abs(intervention_penalty):
        raise ValueError(
            f"HANDBACK_REWARD magnitude ({abs(handback_reward)}) must not exceed "
            f"INTERVENTION_PENALTY magnitude ({abs(intervention_penalty)}) -- otherwise a human "
            "handing back control could outweigh the penalty for making them intervene in the first place."
        )


validate_reward_magnitudes(HANDBACK_REWARD, INTERVENTION_PENALTY)

# See module docstring: reward/termination applied to the pending transition
# when an IK solve fails while computing a human-intervention override.
IK_FAILURE_PENALTY = -1.0

# See module docstring's "Delta rather than absolute arm control" section.
# Max normalized ([-1, 1]-space) position change the policy can command per
# arm joint, per tick -- TODO: dial in against how it actually feels/trains.
ARM_MAX_DELTA_PER_TICK = 0.05

AVERAGE_OVER_EPISODES = 10


@dataclass
class PendingTransition:
    """One store_transition() call's worth of arguments, held back for a tick
    -- see PENALIZE_ACTION_BEFORE_INTERVENTION in the module docstring."""

    pre_state: np.ndarray
    action: np.ndarray
    reward: float
    post_state: np.ndarray
    done: bool
    override_action: np.ndarray | None


def relabel_pending_if_intervention_started(
    pending: PendingTransition | None,
    intervening: bool,
    was_intervening: bool,
) -> PendingTransition | None:
    """If intervention just started this tick (intervening and not
    was_intervening) and pending is eligible (not None, not already terminal),
    returns pending with its reward replaced by INTERVENTION_PENALTY.
    Otherwise returns pending unchanged. See PENALIZE_ACTION_BEFORE_INTERVENTION
    in the module docstring."""
    if (
        PENALIZE_ACTION_BEFORE_INTERVENTION
        and intervening
        and not was_intervening
        and pending is not None
        and not pending.done
    ):
        return replace(pending, reward=INTERVENTION_PENALTY)
    return pending


def relabel_pending_if_intervention_ended(
    pending: PendingTransition | None,
    intervening: bool,
    was_intervening: bool,
) -> PendingTransition | None:
    """If intervention just ended this tick (not intervening and was_intervening)
    and pending is eligible (not None, not already terminal), returns pending
    with its reward replaced by HANDBACK_REWARD. Otherwise returns pending
    unchanged. See REWARD_ACTION_BEFORE_HANDBACK in the module docstring."""
    if (
        REWARD_ACTION_BEFORE_HANDBACK
        and not intervening
        and was_intervening
        and pending is not None
        and not pending.done
    ):
        return replace(pending, reward=HANDBACK_REWARD)
    return pending


def episode_return_delta_for_relabel(old_reward: float | None, relabeled: PendingTransition | None) -> float:
    """The amount a running episode_return must be adjusted by after a relabel
    (see relabel_pending_if_intervention_started / mark_pending_as_terminal_failure)
    -- episode_return already added pending's *original* reward a tick ago, so
    if relabeling changed it, only the delta needs adding now, not the full
    new reward."""
    if relabeled is not None and old_reward is not None and relabeled.reward != old_reward:
        return relabeled.reward - old_reward
    return 0.0


def mark_pending_as_terminal_failure(
    pending: PendingTransition | None,
    penalty: float,
) -> PendingTransition | None:
    """Relabels pending (if any, and not already terminal) as a hard failure --
    reward set to penalty, done set to True -- and returns it. Otherwise
    returns pending unchanged. Used when an IK solve fails while computing a
    human-intervention override: pending is whichever action actually left the
    arm in its current, unrecoverable pose, so that's exactly what should be
    penalized and the episode ended, mirroring compute_reward's own terminal
    cases. Never overwrites an already-terminal pending transition, for the
    same reason as relabel_pending_if_intervention_started."""
    if pending is None or pending.done:
        return pending
    return replace(pending, reward=penalty, done=True)


def delta_action_to_absolute(
    delta_action: np.ndarray, observation: Observation, bounds: list[tuple[float, float]]
) -> np.ndarray:
    """Converts a delta-space action -- a normalized ([-1, 1]) change in
    position for each of the 6 arm joints, scaled by ARM_MAX_DELTA_PER_TICK,
    plus an absolute gripper command in the last slot, unchanged -- into the
    absolute normalized targets Unity's wire protocol actually expects (see
    remote_connection.py). See "Delta rather than absolute arm control" in the
    module docstring."""
    current_normalized = ik.angles_to_normalized(observation.joint_positions_radians[: rc.NUM_ARM_JOINTS], bounds)
    arm_target = np.clip(current_normalized + delta_action[: rc.NUM_ARM_JOINTS] * ARM_MAX_DELTA_PER_TICK, -1.0, 1.0)
    return np.concatenate([arm_target, delta_action[rc.NUM_ARM_JOINTS:]])


def absolute_action_to_delta(
    absolute_action: np.ndarray, observation: Observation, bounds: list[tuple[float, float]]
) -> np.ndarray:
    """Inverse of delta_action_to_absolute -- converts an absolute arm target
    (e.g. a human intervention's IK-solved override_action) into the
    equivalent delta-space action relative to observation's current joint
    positions, clipped to [-1, 1] (i.e. to at most ARM_MAX_DELTA_PER_TICK of
    normalized movement). Used so a human's intervention is stored in the same
    action space the policy predicts in, keeping the replay buffer's action
    semantics consistent regardless of who actually executed it -- note that
    what's clipped here is only what's *stored* for learning, not what's
    actually sent to Unity: a human's own control isn't rate-limited by the
    policy's step size."""
    current_normalized = ik.angles_to_normalized(observation.joint_positions_radians[: rc.NUM_ARM_JOINTS], bounds)
    arm_delta = np.clip(
        (absolute_action[: rc.NUM_ARM_JOINTS] - current_normalized) / ARM_MAX_DELTA_PER_TICK, -1.0, 1.0
    )
    return np.concatenate([arm_delta, absolute_action[rc.NUM_ARM_JOINTS:]])


def compute_reward(placement_state: PlacementState, block_height_above_table: float) -> tuple[float, bool]:
    """+1 on success, -1 on falling off the table (both terminal -- done is
    True exactly here, since Unity has already scheduled a block reset for
    immediately after this tick). Otherwise dense shaping from
    block_height_above_table (the block's current height minus its
    per-episode table/spawn reference, see main()): RESTING_ON_TABLE_REWARD
    while it's within TABLE_REST_TOLERANCE_METERS of that reference (i.e.
    still sitting untouched), IN_AIR_REWARD once it's been lifted higher than
    that. See module docstring's opening "Reward" paragraph."""
    if placement_state == PlacementState.INSIDE_PLACED:
        return SUCCESS_REWARD, True
    if placement_state == PlacementState.FAILED_FELL:
        return FAILURE_REWARD, True
    if block_height_above_table <= TABLE_REST_TOLERANCE_METERS:
        return RESTING_ON_TABLE_REWARD, False
    return IN_AIR_REWARD, False


class InterventionController:
    """Spherical-coordinate teleop control (see teleop_controller.py's module
    docstring for the control scheme itself), gated behind an explicit "t"
    toggle rather than momentary key-holding: pressing "t" flips between
    teleop mode (this class computes every override action; the autonomous
    policy is completely ignored, even on ticks with no key held -- the arm
    just holds position, same as teleop_controller.py itself) and policy mode
    (this class returns no override at all, WASD/Up/Down/Space do nothing).
    Holding movement keys down was the original design, but made it hard to
    sustain a deliberate, multi-step intervention: releasing a key even
    briefly (e.g. moving from W to A) handed control back to the policy for
    that tick, causing jerky, flickering control.

    The (r, theta, phi) target and IK seed are internal state, carried forward
    tick-to-tick like teleop_controller.py's own loop -- but only *within* a
    teleop session. They're re-derived from the live observation exactly once,
    on the tick teleop mode is entered, so a handoff from the policy always
    picks up smoothly from wherever the arm actually is rather than jumping to
    stale state left over from a previous session.
    """

    def __init__(self, chain, bounds: list[tuple[float, float]]) -> None:
        self._chain = chain
        self._bounds = bounds
        self._gripper_open = False
        self._gripper_command = GRIPPER_CLOSED_COMMAND
        self._teleop_mode = False
        self._r = 0.0
        self._theta = 0.0
        self._phi = 0.0
        self._previous_angles = np.zeros(ik.NUM_JOINTS)

    def compute(
        self, observation: Observation, pressed_keys, teleop_toggled: bool, gripper_toggled: bool
    ) -> tuple[np.ndarray | None, bool]:
        """Returns (override_action, intervening); override_action is only
        meaningful when intervening is True."""
        if teleop_toggled:
            self._teleop_mode = not self._teleop_mode
            if self._teleop_mode:
                eef_position_flu = ik.ruf_to_flu(observation.end_effector_position_ruf)
                self._r, self._theta, self._phi = cartesian_to_spherical(eef_position_flu)
                self._previous_angles = np.concatenate(
                    [observation.joint_positions_radians[:NUM_POSITION_JOINTS], WRIST_JOINT_ANGLES]
                )

        if not self._teleop_mode:
            return None, False

        if gripper_toggled:
            self._gripper_open = not self._gripper_open
        gripper_target = GRIPPER_OPEN_COMMAND if self._gripper_open else GRIPPER_CLOSED_COMMAND
        self._gripper_command = step_toward(self._gripper_command, gripper_target, GRIPPER_STEP_PER_TICK)

        dr, dtheta, dphi = compute_spherical_delta(pressed_keys)
        self._r = np.clip(self._r + dr, R_MIN_METERS, R_MAX_METERS)
        self._theta = self._theta + dtheta
        self._phi = np.clip(self._phi + dphi, PHI_MIN_RADIANS, PHI_MAX_RADIANS)
        target_position_flu = spherical_to_cartesian(self._r, self._theta, self._phi)

        angles = ik.solve_ik(self._chain, target_position_flu, initial_angles=self._previous_angles)
        self._previous_angles = angles
        normalized_arm_actions = ik.angles_to_normalized(angles, self._bounds)
        override_action = np.concatenate([normalized_arm_actions, [self._gripper_command]])
        return override_action, True


def main() -> None:
    print("Loading IK chain...")
    trimmed_urdf_path = ik.trim_urdf_subtree(str(URDF_PATH), GRIPPER_ROOT_LINK)
    chain = ik.load_arm_chain(trimmed_urdf_path, num_active_joints=NUM_POSITION_JOINTS)
    bounds = ik.joint_bounds(chain)
    intervention = InterventionController(chain, bounds)

    policy = RLPDPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    DEFAULT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = rc.connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    print("Connected.")

    pygame.init()
    pygame.display.set_mode((480, 160))
    pygame.display.set_caption("RLPD training -- 't' toggles teleop, WASD/Up/Down/Space drive while teleop is on")
    clock = pygame.time.Clock()

    print(__doc__)

    episode_return = 0.0
    episode_returns: list[float] = []
    num_successes = 0
    num_ik_failures = 0
    was_intervening = False
    pending: PendingTransition | None = None

    def flush_pending() -> None:
        if pending is not None:
            policy.store_transition(
                pending.pre_state,
                pending.action,
                pending.reward,
                pending.post_state,
                pending.done,
                override_action=pending.override_action,
            )

    def adjust_episode_return_for_relabel(old_reward: float | None, relabeled: PendingTransition | None) -> None:
        nonlocal episode_return
        episode_return += episode_return_delta_for_relabel(old_reward, relabeled)

    def announce_relabel_if_changed(
        relabeled_tick: int, old_reward: float | None, relabeled: PendingTransition | None, reason: str
    ) -> None:
        # The tick that actually gets relabeled already printed its own log
        # line last iteration, showing its *original* reward -- without this,
        # a relabel is invisible: nothing in the log ever shows the corrected
        # value it was actually stored with.
        if relabeled is not None and old_reward is not None and relabeled.reward != old_reward:
            print(f"  -> relabeling tick {relabeled_tick}'s reward {old_reward:+.2f} -> {relabeled.reward:+.2f} ({reason})")

    def record_episode_end(success: bool) -> None:
        nonlocal episode_return, num_successes
        episode_returns.append(episode_return)
        if success:
            num_successes += 1
        episode_return = 0.0

        recent = episode_returns[-AVERAGE_OVER_EPISODES:]
        print(
            f"episode {len(episode_returns):4d}  "
            f"successes: {num_successes}/{len(episode_returns)}  "
            f"ik failures: {num_ik_failures}  "
            f"avg return (last {len(recent)}): {np.mean(recent):+.2f}"
        )
        if len(episode_returns) % CHECKPOINT_EVERY_EPISODES == 0:
            policy.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
            print(f"Saved checkpoint to {DEFAULT_CHECKPOINT_PATH}")

    try:
        with sock:
            # Bootstrap: send a neutral (closed-gripper, zero-target) action to
            # get the first observation, same as random_controller.py/teleop_controller.py.
            neutral_action = np.zeros(rc.NUM_ARM_JOINTS)
            rc.send_action(sock, neutral_action, GRIPPER_CLOSED_COMMAND)
            observation = rc.read_observation(sock)
            pre_state = rc.observation_to_array(observation)
            # RUF's up axis (see ik.py) -- per-episode reference height for
            # compute_reward's resting-on-table/in-air shaping, re-captured
            # from the first observation of every fresh episode (below).
            table_height = observation.block_position_ruf[1]

            tick = 0
            running = True
            pending_reset = False
            while running:
                teleop_toggled = False
                gripper_toggled = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                        teleop_toggled = True
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        gripper_toggled = True

                pressed_keys = pygame.key.get_pressed()
                action = policy.act(pre_state)
                q_estimate = policy.estimate_q(pre_state, action)

                try:
                    override_action, intervening = intervention.compute(
                        observation, pressed_keys, teleop_toggled, gripper_toggled
                    )
                except ValueError as e:
                    # See module docstring's "IK solve failures" section:
                    # pending is whichever action left the arm in this
                    # unrecoverable pose -- relabel it as a hard failure, then
                    # ask Unity to reset the block *and* the arm.
                    print(f"\nIK solve failed ({e}) -- treating as failure, resetting arm.")
                    num_ik_failures += 1
                    old_reward = pending.reward if pending is not None else None
                    pending = mark_pending_as_terminal_failure(pending, IK_FAILURE_PENALTY)
                    adjust_episode_return_for_relabel(old_reward, pending)
                    announce_relabel_if_changed(tick - 1, old_reward, pending, "IK solve failure")
                    flush_pending()
                    pending = None
                    record_episode_end(success=False)

                    rc.send_action(sock, np.zeros(rc.NUM_ARM_JOINTS), GRIPPER_CLOSED_COMMAND, reset=True)
                    observation = rc.read_observation(sock)
                    pre_state = rc.observation_to_array(observation)
                    table_height = observation.block_position_ruf[1]
                    pending_reset = False
                    was_intervening = False
                    clock.tick(60)
                    continue

                if intervening:
                    executed_action = override_action
                    stored_override_action = absolute_action_to_delta(override_action, observation, bounds)
                else:
                    executed_action = delta_action_to_absolute(action, observation, bounds)
                    stored_override_action = None

                # The previous tick's transition is only flushed now, once
                # this tick reveals whether an intervention boundary (start or
                # end) just happened -- see PENALIZE_ACTION_BEFORE_INTERVENTION
                # / REWARD_ACTION_BEFORE_HANDBACK in the module docstring. The
                # two relabels are mutually exclusive (intervening can only
                # flip one direction per tick), so chaining them is safe.
                old_reward = pending.reward if pending is not None else None
                pending = relabel_pending_if_intervention_started(pending, intervening, was_intervening)
                pending = relabel_pending_if_intervention_ended(pending, intervening, was_intervening)
                adjust_episode_return_for_relabel(old_reward, pending)
                if intervening and not was_intervening:
                    relabel_reason = "human intervention started"
                elif was_intervening and not intervening:
                    relabel_reason = "human handed control back to policy"
                else:
                    relabel_reason = ""
                announce_relabel_if_changed(tick - 1, old_reward, pending, relabel_reason)
                flush_pending()

                rc.send_action(sock, executed_action[: rc.NUM_ARM_JOINTS], executed_action[rc.NUM_ARM_JOINTS])
                observation = rc.read_observation(sock)
                post_state = rc.observation_to_array(observation)

                if pending_reset:
                    # First observation of a fresh episode -- re-anchor the
                    # table-height reference compute_reward's shaping uses.
                    table_height = observation.block_position_ruf[1]
                    pending_reset = False

                block_height_above_table = observation.block_position_ruf[1] - table_height
                reward, done = compute_reward(observation.placement_state, block_height_above_table)
                pending = PendingTransition(
                    pre_state, action, reward, post_state, done, stored_override_action
                )
                was_intervening = intervening
                episode_return += reward
                pre_state = post_state

                source = "human" if intervening else "policy"
                action_str = np.array2string(executed_action, precision=2, floatmode="fixed", suppress_small=True)
                print(
                    f"tick={tick:6d} placement={observation.placement_state.name:15s} "
                    f"source={source:6s} Q={q_estimate:+7.3f} reward={reward:+.2f} action={action_str}"
                )
                tick += 1

                if done:
                    pending_reset = True
                    record_episode_end(success=observation.placement_state == PlacementState.INSIDE_PLACED)

                clock.tick(60)
    except (KeyboardInterrupt, OSError) as e:
        print(f"\nStopping ({e}).")
    finally:
        pygame.quit()

    flush_pending()
    policy.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
    print(f"Saved final checkpoint to {DEFAULT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
