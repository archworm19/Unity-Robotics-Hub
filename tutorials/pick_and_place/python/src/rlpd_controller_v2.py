"""Human-in-the-loop RLPD training against the live Unity task (see policy.py's
HILPolicy and rlpd_policy.RLPDPolicy). Rewrite of rlpd_controller.py, which
accumulated too much timing/state complexity (a whole PendingTransition +
retroactive-relabeling mechanism) trying to work around not knowing whether a
human had just started intervening until a tick later.

Loop, once per TICK_INTERVAL_SECONDS:
  1. send the action decided last iteration (converted to a joint target --
     see "policy operates in human space" below); read back the resulting
     state and compute its raw reward (see compute_reward)
  2. wait TICK_INTERVAL_SECONDS, continuously sampling keyboard state the
     whole time (see wait_and_sample_input) -- the human gets this whole
     window to react to the state observed in step 1, and to press "t" if
     they want to toggle control
  3. if "t" was pressed, flip human_control_state. If that flip just handed
     control *to* a human, add INTERVENTION_PENALTY to the reward; if it just
     handed control *back to* the policy, add HANDBACK_REWARD instead
  4. store the (state, action, reward, next_state, done) transition
  5. decide the *next* action -- from the human's held keys
     (human_action_from_keys) if human_control_state is on, otherwise from
     the policy -- and loop

Because the human's reaction is awaited *before* deciding a transition's
reward, there's no need to hold anything back or retroactively relabel it the
way rlpd_controller.py did -- intervention boundaries are known synchronously.

Control is toggled explicitly by "t", not inferred from whether a key happens
to be held: an earlier version of this loop treated any held W/S/A/D/Up/Down/
O/P as "the human is in control this tick," which in practice caused
unwanted flicker back to the policy during brief, ordinary pauses (e.g.
releasing W a moment before pressing A, or just pausing to look at the scene)
-- not something a human actually intends as "give control back." "t" makes
handoff a deliberate act in both directions: press it once to take over,
press it again to hand back. While human_control_state is on and no key is
held, the arm simply holds its current pose rather than reverting to the
policy.

"u" forces an immediate, episode-ending reset (FORCE_RESET_PENALTY) for
situations neither Unity's own success/fall detection nor an IK failure
catches -- e.g. the block wedged against the arm or the edge of the goal
zone. Unlike a natural success/fail, Unity never resets on its own for this,
so it's requested explicitly (block *and* arm, same as an IK-failure reset),
and control always reverts to the policy afterward.

Episode boundaries are entirely Unity-driven: RemoteJointController resets the
block automatically once placement_state comes back INSIDE_PLACED (success) or
FAILED_FELL (fell off the table) -- see remote_connection.py's module
docstring. Reward is +1 on success, -1 on falling off the table (both
terminal), otherwise dense shaping from the block's height above its
per-episode table/spawn reference (RESTING_ON_TABLE_REWARD vs IN_AIR_REWARD)
plus a continuous pull toward both reaching the block and carrying it to the
goal (BLOCK_TO_GOAL_DISTANCE_PENALTY_PER_METER and
GRIPPER_TO_BLOCK_DISTANCE_PENALTY_PER_METER -- see compute_reward).

The policy operates in human space: its action isn't an absolute joint target,
it's the exact same 4-dim (r, theta, phi, gripper) pose-delta a human's held
keys produce (see human_action_from_keys) -- W/S, A/D, Up/Down, and O/P for
the gripper (O = toward open, P = toward closed), each dimension in [-1, 1]
representing a fraction of one tick's max step. pose_delta_to_joint_action is
the single conversion from that shared action space into the absolute
normalized joint targets Unity's wire protocol actually expects (unchanged)
-- both a human's held keys and the policy's own output go through it,
mutating the same PoseState, so there's no separate delta-vs-absolute
bookkeeping needed for storage the way rlpd_controller.py needed: whichever
action was actually executed (policy's or a human's override) is already in
the policy's own native action space, ready to store as-is. The step sizes
(POSE_STEP_*) are this module's own, not teleop_controller.py's -- reusing
its 60 Hz-tuned constants directly at this module's much slower
TICK_INTERVAL_SECONDS cadence would make every control input feel ~10x
sluggish.

Because PoseState is continuously driven by whoever's in control every single
tick (never left idle), there's also no "stale state from a previous
intervention" problem to resync away -- IK is always seeded from its own last
successful solve, not from potentially-noisy live joint angles, so IK solve
failures (an untrained policy walking the shared pose to something ikpy's
solver rejects as unreachable) should be rare. When one does happen, it isn't
retroactively penalized -- by the time it's discovered, whatever transition
led here is already stored with whatever reward it naturally got. It just
logs, asks Unity to reset the block *and* the arm (see
remote_connection.send_action's reset flag), reseeds PoseState from the fresh
observation, and resumes under policy control.

All logging goes to the pygame window (see LogWindow) -- there's no terminal
output once it's running, since the window needs focus for keyboard input
anyway.
"""

import sys
import time
from collections import deque
from dataclasses import dataclass
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
    NUM_POSITION_JOINTS,
    PHI_MAX_RADIANS,
    PHI_MIN_RADIANS,
    R_MAX_METERS,
    R_MIN_METERS,
    URDF_PATH,
    WRIST_JOINT_ANGLES,
    cartesian_to_spherical,
    spherical_to_cartesian,
)

HOST = "127.0.0.1"
PORT = 9000
CONNECT_TIMEOUT_SECONDS = 30.0

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
DEFAULT_CHECKPOINT_PATH = TELEOP_DIRECTORY / "rlpd.pt"
CHECKPOINT_EVERY_EPISODES = 10

STATE_DIM = rc.NUM_OBSERVATION_FLOATS
POLICY_ACTION_DIM = 4  # (r, theta, phi, gripper) pose-delta -- see module docstring

SUCCESS_REWARD = 5.0
FAILURE_REWARD = -5.0

# Dense shaping applied on every non-terminal tick, from the block's height
# above its per-episode table/spawn reference (see main()'s table_height).
TABLE_REST_TOLERANCE_METERS = 0.02  # how far above table height still counts as "resting", not lifted
RESTING_ON_TABLE_REWARD = -1.0
IN_AIR_REWARD = 0.0

# Additional dense shaping applied on every non-terminal tick: penalties
# proportional to (a) how far the block currently is from the goal zone, and
# (b) how far the gripper currently is from the block -- directional signal
# to both reach for the block and carry it to the goal, not just to lift it
# (RESTING_ON_TABLE_REWARD/IN_AIR_REWARD above) or the sparse success/failure
# signal. Both negative, scaled small like the other dense terms so they
# nudge rather than dominate.
BLOCK_TO_GOAL_DISTANCE_PENALTY_PER_METER = -1.0
GRIPPER_TO_BLOCK_DISTANCE_PENALTY_PER_METER = -1.0

# Larger in magnitude than a plain failure -- a human having to step in is a
# stronger negative signal than just letting the episode fail and reset.
INTERVENTION_PENALTY = 0.0
# The human's last action right before handing control back to the policy --
# the state they chose to hand back from is exactly what the policy should
# learn to reach and continue from on its own.
HANDBACK_REWARD = 0.0

# "u" forces an immediate episode-ending reset (block AND arm, same as an
# IK-failure reset) when a human judges the current state unrecoverable --
# e.g. the block wedged somewhere neither placement nor fall detection
# catches. As severe as a plain failure, since that's effectively what it is.
FORCE_RESET_PENALTY = -5.0

if abs(HANDBACK_REWARD) > abs(INTERVENTION_PENALTY):
    raise ValueError(
        f"HANDBACK_REWARD magnitude ({abs(HANDBACK_REWARD)}) must not exceed "
        f"INTERVENTION_PENALTY magnitude ({abs(INTERVENTION_PENALTY)}) -- otherwise a human "
        "handing back control could outweigh the penalty for making them intervene in the first place."
    )

# How long each loop iteration waits for a human reaction, continuously
# sampling keyboard state the whole time (see wait_and_sample_input) -- also
# how much wall-clock time separates consecutive stored transitions.
TICK_INTERVAL_SECONDS = 0.2
POLL_INTERVAL_SECONDS = 0.02  # sampling granularity within that wait

# This module's own per-tick step sizes for the shared (r, theta, phi,
# gripper) pose-delta action space -- see module docstring for why these are
# separate from teleop_controller.py's own, 60 Hz-tuned constants.
POSE_STEP_METERS_PER_TICK = 0.01
POSE_STEP_RADIANS_PER_TICK = np.radians(4.0)
POSE_GRIPPER_STEP_PER_TICK = 0.25  # fraction of full open<->closed range per tick

AVERAGE_OVER_EPISODES = 10

LOG_WINDOW_SIZE = (950, 420)
LOG_MAX_LINES = 22
LOG_FONT_SIZE = 13
LOG_BACKGROUND_COLOR = (24, 24, 24)
LOG_TEXT_COLOR = (220, 220, 220)


class LogWindow:
    """The only place this module writes output -- a small scrolling console
    rendered directly into the pygame window, since that window needs to stay
    focused to receive keyboard input for human intervention anyway."""

    def __init__(self) -> None:
        pygame.font.init()
        self._font = pygame.font.SysFont("menlo,consolas,monospace", LOG_FONT_SIZE)
        self._surface = pygame.display.set_mode(LOG_WINDOW_SIZE)
        pygame.display.set_caption(
            "RLPD training -- 't' toggles control, 'u' forces a reset; WASD/Up/Down/O/P drive while you're in control"
        )
        self._lines: deque[str] = deque(maxlen=LOG_MAX_LINES)
        self._redraw()

    def log(self, line: str) -> None:
        for sub_line in line.splitlines() or [""]:
            self._lines.append(sub_line)
        self._redraw()

    def _redraw(self) -> None:
        self._surface.fill(LOG_BACKGROUND_COLOR)
        for i, line in enumerate(self._lines):
            rendered = self._font.render(line, True, LOG_TEXT_COLOR)
            self._surface.blit(rendered, (8, 8 + i * (LOG_FONT_SIZE + 4)))
        pygame.display.flip()


def compute_reward(
    placement_state: PlacementState,
    block_height_above_table: float,
    block_to_goal_distance: float,
    gripper_to_block_distance: float,
) -> tuple[float, bool]:
    """+1 on success, -1 on falling off the table (both terminal, ignoring
    the distances -- placement_state already says everything that matters
    once the episode is over). Otherwise dense shaping: RESTING_ON_TABLE_REWARD
    while the block is within TABLE_REST_TOLERANCE_METERS of its per-episode
    table/spawn reference, IN_AIR_REWARD once lifted higher than that, plus
    BLOCK_TO_GOAL_DISTANCE_PENALTY_PER_METER * block_to_goal_distance and
    GRIPPER_TO_BLOCK_DISTANCE_PENALTY_PER_METER * gripper_to_block_distance
    either way, so there's continuous pull toward both reaching the block and
    carrying it to the goal, on top of the height-based term."""
    if placement_state == PlacementState.INSIDE_PLACED:
        return SUCCESS_REWARD, True
    if placement_state == PlacementState.FAILED_FELL:
        return FAILURE_REWARD, True
    distance_reward = (
        BLOCK_TO_GOAL_DISTANCE_PENALTY_PER_METER * block_to_goal_distance
        + GRIPPER_TO_BLOCK_DISTANCE_PENALTY_PER_METER * gripper_to_block_distance
    )
    if block_height_above_table <= TABLE_REST_TOLERANCE_METERS:
        return RESTING_ON_TABLE_REWARD + distance_reward, False
    return IN_AIR_REWARD + distance_reward, False


def wait_and_sample_input(duration_seconds: float) -> tuple[object, bool, bool, bool]:
    """Continuously polls pygame's event queue and key state for
    duration_seconds. Returns (pressed_keys, toggle_requested,
    force_reset_requested, quit_requested): pressed_keys is the *last* sample
    taken, right at the end of the window; toggle_requested is True if "t"
    was pressed (KEYDOWN) at any point during the window, and
    force_reset_requested likewise for "u" -- both edge-triggered, so each
    fires once per press, not once per tick the key happens to still be
    held."""
    deadline = time.monotonic() + duration_seconds
    toggle_requested = False
    force_reset_requested = False
    quit_requested = False
    pressed_keys = pygame.key.get_pressed()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                toggle_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                force_reset_requested = True
        pressed_keys = pygame.key.get_pressed()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))
    return pressed_keys, toggle_requested, force_reset_requested, quit_requested


def human_action_from_keys(pressed_keys) -> np.ndarray:
    """4-dim (r, theta, phi, gripper) action in the exact same [-1, 1]-per-
    dimension space the policy outputs -- +1/-1 for whichever key (if any) is
    held for that dimension, 0 otherwise: W/S, A/D, Up/Down (same mapping as
    teleop_controller.py's compute_spherical_delta), and O/P for the gripper
    (O = toward open, P = toward closed)."""
    r = 0.0
    if pressed_keys[pygame.K_w]:
        r += 1.0
    if pressed_keys[pygame.K_s]:
        r -= 1.0
    theta = 0.0
    if pressed_keys[pygame.K_a]:
        theta += 1.0
    if pressed_keys[pygame.K_d]:
        theta -= 1.0
    phi = 0.0
    if pressed_keys[pygame.K_UP]:
        phi -= 1.0
    if pressed_keys[pygame.K_DOWN]:
        phi += 1.0
    gripper = 0.0
    if pressed_keys[pygame.K_o]:
        gripper += 1.0
    if pressed_keys[pygame.K_p]:
        gripper -= 1.0
    return np.array([r, theta, phi, gripper])


@dataclass
class PoseState:
    """Shared spherical-coordinate target (centered on base_link) and IK seed
    -- mutated by pose_delta_to_joint_action regardless of whether the delta
    driving it came from a human's held keys or the policy's own action."""

    r: float
    theta: float
    phi: float
    gripper_command: float
    previous_angles: np.ndarray


def seed_pose_from_observation(observation: Observation) -> PoseState:
    """Builds a fresh PoseState anchored to the arm's actual current pose --
    used once at startup and again after an IK-failure reset, so the shared
    pose state never starts out (or resumes) somewhere the arm isn't."""
    eef_position_flu = ik.ruf_to_flu(observation.end_effector_position_ruf)
    r, theta, phi = cartesian_to_spherical(eef_position_flu)
    return PoseState(
        r=r,
        theta=theta,
        phi=phi,
        gripper_command=GRIPPER_CLOSED_COMMAND,
        previous_angles=np.concatenate(
            [observation.joint_positions_radians[:NUM_POSITION_JOINTS], WRIST_JOINT_ANGLES]
        ),
    )


def pose_delta_to_joint_action(
    chain, bounds: list[tuple[float, float]], pose: PoseState, action: np.ndarray
) -> np.ndarray:
    """Scales a 4-dim (r, theta, phi, gripper) action by this module's
    POSE_STEP_* constants, applies it to pose (mutated in place), and returns
    the resulting absolute normalized joint-space action Unity's wire
    protocol expects (see remote_connection.py) -- the single conversion both
    a human's held keys and the policy's own action go through.

    action is clipped to [-1, 1] first: the policy's own output is already
    continuous within that range (GaussianPolicy tanh-squashes it), and this
    makes that bound explicit and guaranteed rather than relying on it
    implicitly -- the policy can move as smoothly and as slowly as it likes,
    but per tick, never faster than a human's own held key (magnitude 1)
    would.

    Can raise ValueError if the resulting target is unreachable (see module
    docstring's "IK solve failures" discussion)."""
    action = np.clip(action, -1.0, 1.0)
    dr = float(action[0]) * POSE_STEP_METERS_PER_TICK
    dtheta = float(action[1]) * POSE_STEP_RADIANS_PER_TICK
    dphi = float(action[2]) * POSE_STEP_RADIANS_PER_TICK
    d_gripper = float(action[3]) * POSE_GRIPPER_STEP_PER_TICK

    pose.r = float(np.clip(pose.r + dr, R_MIN_METERS, R_MAX_METERS))
    pose.theta = pose.theta + dtheta
    pose.phi = float(np.clip(pose.phi + dphi, PHI_MIN_RADIANS, PHI_MAX_RADIANS))
    pose.gripper_command = float(
        np.clip(pose.gripper_command + d_gripper, GRIPPER_CLOSED_COMMAND, GRIPPER_OPEN_COMMAND)
    )

    target_position_flu = spherical_to_cartesian(pose.r, pose.theta, pose.phi)
    angles = ik.solve_ik(chain, target_position_flu, initial_angles=pose.previous_angles)
    pose.previous_angles = angles
    normalized_arm_actions = ik.angles_to_normalized(angles, bounds)
    return np.concatenate([normalized_arm_actions, [pose.gripper_command]])


def main() -> None:
    pygame.init()
    log = LogWindow()

    log.log("Loading IK chain...")
    trimmed_urdf_path = ik.trim_urdf_subtree(str(URDF_PATH), GRIPPER_ROOT_LINK)
    chain = ik.load_arm_chain(trimmed_urdf_path, num_active_joints=NUM_POSITION_JOINTS)
    bounds = ik.joint_bounds(chain)

    policy = RLPDPolicy(state_dim=STATE_DIM, action_dim=POLICY_ACTION_DIM)
    DEFAULT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.log(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = rc.connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    log.log("Connected.")

    episode_return = 0.0
    episode_returns: list[float] = []
    num_successes = 0
    num_ik_failures = 0

    def record_episode_end(success: bool) -> None:
        nonlocal episode_return, num_successes
        episode_returns.append(episode_return)
        if success:
            num_successes += 1
        episode_return = 0.0
        recent = episode_returns[-AVERAGE_OVER_EPISODES:]
        log.log(
            f"episode {len(episode_returns):4d}  successes: {num_successes}/{len(episode_returns)}  "
            f"ik failures: {num_ik_failures}  avg return (last {len(recent)}): {np.mean(recent):+.2f}"
        )
        if len(episode_returns) % CHECKPOINT_EVERY_EPISODES == 0:
            policy.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
            log.log(f"Saved checkpoint to {DEFAULT_CHECKPOINT_PATH}")

    try:
        with sock:
            neutral = np.concatenate([np.zeros(rc.NUM_ARM_JOINTS), [GRIPPER_CLOSED_COMMAND]])
            rc.send_action(sock, neutral[: rc.NUM_ARM_JOINTS], neutral[rc.NUM_ARM_JOINTS])
            observation = rc.read_observation(sock)
            state = rc.observation_to_array(observation)
            # RUF's up axis (see ik.py) -- per-episode reference height for
            # compute_reward's resting-on-table/in-air shaping, re-anchored
            # from the first observation of every fresh episode (below).
            table_height = observation.block_position_ruf[1]
            pose = seed_pose_from_observation(observation)

            human_control_state = False
            action = policy.act(state)  # 4-dim, human-space (see module docstring)
            pending_reset = False
            tick = 0

            while True:
                try:
                    joint_action = pose_delta_to_joint_action(chain, bounds, pose, action)
                except ValueError as e:
                    log.log(f"IK solve failed ({e}) -- resetting arm, resuming under policy control.")
                    num_ik_failures += 1
                    rc.send_action(sock, np.zeros(rc.NUM_ARM_JOINTS), GRIPPER_CLOSED_COMMAND, reset=True)
                    observation = rc.read_observation(sock)
                    state = rc.observation_to_array(observation)
                    table_height = observation.block_position_ruf[1]
                    pose = seed_pose_from_observation(observation)
                    pending_reset = False
                    human_control_state = False
                    action = policy.act(state)
                    continue

                rc.send_action(sock, joint_action[: rc.NUM_ARM_JOINTS], joint_action[rc.NUM_ARM_JOINTS])
                observation = rc.read_observation(sock)
                next_state = rc.observation_to_array(observation)

                if pending_reset:
                    table_height = observation.block_position_ruf[1]
                    pending_reset = False

                block_height_above_table = observation.block_position_ruf[1] - table_height
                gripper_height_above_table = observation.end_effector_position_ruf[1] - table_height
                block_to_goal_distance = float(
                    np.linalg.norm(observation.block_position_ruf - observation.goal_position_ruf)
                )
                gripper_to_block_distance = float(
                    np.linalg.norm(observation.end_effector_position_ruf - observation.block_position_ruf)
                )
                raw_reward, done = compute_reward(
                    observation.placement_state,
                    block_height_above_table,
                    block_to_goal_distance,
                    gripper_to_block_distance,
                )

                pressed_keys, toggle_requested, force_reset_requested, quit_requested = wait_and_sample_input(
                    TICK_INTERVAL_SECONDS
                )
                if quit_requested:
                    break

                if force_reset_requested:
                    reward = FORCE_RESET_PENALTY
                    done = True
                    log.log(f"  -> force reset requested (reward={reward:+.2f})")
                else:
                    was_human_control = human_control_state
                    if toggle_requested:
                        human_control_state = not human_control_state

                    if human_control_state and not was_human_control:
                        reward = raw_reward + INTERVENTION_PENALTY
                        log.log(f"  -> human took control (reward {raw_reward:+.2f} -> {reward:+.2f})")
                    elif was_human_control and not human_control_state:
                        reward = raw_reward + HANDBACK_REWARD
                        log.log(f"  -> handing back to policy (reward {raw_reward:+.2f} -> {reward:+.2f})")
                    else:
                        reward = raw_reward

                human_action = human_action_from_keys(pressed_keys)
                q_estimate = policy.estimate_q(state, action)

                policy.store_transition(
                    state, action, reward, next_state, done, override_action=action if human_control_state else None
                )
                episode_return += reward

                log.log(
                    f"tick={tick:6d} placement={observation.placement_state.name:15s} "
                    f"source={'human' if human_control_state else 'policy':6s} Q={q_estimate:+7.3f} "
                    f"gripper_h={gripper_height_above_table:+.3f} reward={reward:+.2f}"
                )
                tick += 1

                if force_reset_requested:
                    # Unlike a natural success/fail, Unity never resets for
                    # this on its own -- request it explicitly (block AND
                    # arm), same as the IK-failure recovery above.
                    record_episode_end(success=False)
                    rc.send_action(sock, np.zeros(rc.NUM_ARM_JOINTS), GRIPPER_CLOSED_COMMAND, reset=True)
                    observation = rc.read_observation(sock)
                    state = rc.observation_to_array(observation)
                    table_height = observation.block_position_ruf[1]
                    pose = seed_pose_from_observation(observation)
                    pending_reset = False
                    human_control_state = False
                    action = policy.act(state)
                    continue

                if done:
                    pending_reset = True
                    record_episode_end(success=observation.placement_state == PlacementState.INSIDE_PLACED)

                state = next_state
                action = human_action if human_control_state else policy.act(state)
    except (KeyboardInterrupt, OSError) as e:
        log.log(f"Stopping ({e}).")
    finally:
        pygame.quit()

    policy.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
    print(f"Saved final checkpoint to {DEFAULT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
