"""Experiment 4: keyboard-driven end-effector translation control.

A small pygame window captures keyboard input (needs OS focus -- click into it,
not the Unity Game view) since terminal input can't give clean "is this key
currently held" state the way a real windowing/event system can. This script
owns all control logic; Unity only ever receives the resulting 7-DOF (6 joint +
gripper) signal and has no concept of keybindings at all -- see
remote_connection.py for the wire protocol.

Translation is controlled in spherical coordinates (r, theta, phi) centered on
base_link, rather than Cartesian xyz -- more intuitive for "reach out and
sweep around" motion than moving along fixed axes:
  W / S        = +r / -r        (move out / in, radially from base_link)
  A / D        = +theta / -theta (counter-clockwise / clockwise around Z, viewed from above)
  Up / Down    = +phi / -phi     (elevation: sweep up towards +Z / down towards horizontal)
  Space        = toggle gripper open/closed

phi is clamped to [0, PHI_MAX_RADIANS] for now (can't sweep below the horizontal
plane). The wrist (joints 4-6) is fixed at WRIST_JOINT_ANGLES -- IK only ever
solves joints 1-3 for position, so the gripper's orientation is never
independently targeted; it's just whatever falls out of wherever the first 3
joints point the arm, rigidly offset by the locked wrist.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pygame

sys.path.insert(0, str(Path(__file__).parent))
import ik
import remote_connection as rc

HOST = "127.0.0.1"
PORT = 9000
CONNECT_TIMEOUT_SECONDS = 30.0

DEFAULT_SAVE_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")

URDF_PATH = Path(__file__).parent / ".." / ".." / "PickAndPlaceProject" / "Assets" / "URDF" / "niryo_one" / "niryo_one.urdf"

# niryo_one.urdf specific: the gripper mechanism isn't part of the arm's own
# 6-DOF chain, and one of its joints has a malformed axis that crashes ikpy's
# parser, so it's pruned before handing the URDF to ikpy.
GRIPPER_ROOT_LINK = "gripper_base"

HOME_POSITION_FLU = np.array([0.245, 0.0, 0.4175])  # arm's home end-effector position

# "Fixed wrist": only joints 1-3 (base yaw, shoulder, elbow) are solved for
# position; joints 4-6 (the spherical wrist -- forearm roll, wrist pitch, hand
# roll) are locked at these angles the whole time, so orientation is never
# independently targeted.
NUM_POSITION_JOINTS = 3
WRIST_JOINT_ANGLES = np.zeros(ik.NUM_JOINTS - NUM_POSITION_JOINTS)

# Reachable workspace, spherical coordinates centered on base_link. The teleop
# target is clamped to this range so holding a key can't walk it out of reach
# and send the IK solver off into a wild, unpredictable pose.
R_MIN_METERS = 0.15
R_MAX_METERS = 0.5
PHI_MIN_RADIANS = 0.0
PHI_MAX_RADIANS = np.radians(50.0)

# TODO: dial in controls with these hyperparams
STEP_METERS_PER_TICK = 0.002
STEP_RADIANS_PER_TICK = np.radians(0.8)
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSED_COMMAND = -1.0
GRIPPER_STEP_PER_TICK = 0.01  # fraction of the full open<->closed range per tick

# Most ticks repeat the same "hold position" action (no key held that tick, and
# the gripper isn't mid-ramp), which would otherwise dominate a recorded
# dataset with redundant near-duplicate frames -- skip recording those by
# default, keeping only ticks where the action actually changed.
RECORD_ONLY_NONZERO_ACTIONS = True


def cartesian_to_spherical(position_flu: npt.ArrayLike) -> tuple[float, float, float]:
    """(r, theta, phi) for a position_flu: r = distance from origin, theta = azimuth
    in the X-Y plane (from +X towards +Y), phi = elevation above the X-Y plane."""
    x, y, z = position_flu
    r = np.linalg.norm(position_flu)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / r)
    return r, theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> npt.NDArray[np.float64]:
    """Inverse of cartesian_to_spherical: (r, theta, phi) -> position_flu."""
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return np.array([x, y, z])


def step_toward(current: float, target: float, step: float) -> float:
    """Move current towards target by at most step, without overshooting."""
    if current < target:
        return min(current + step, target)
    return max(current - step, target)


def compute_spherical_delta(pressed_keys: npt.ArrayLike) -> tuple[float, float, float]:
    """(dr, dtheta, dphi) for one tick from currently held movement keys."""
    dr = 0.0
    dtheta = 0.0
    dphi = 0.0
    if pressed_keys[pygame.K_w]:
        dr += STEP_METERS_PER_TICK
    if pressed_keys[pygame.K_s]:
        dr -= STEP_METERS_PER_TICK
    if pressed_keys[pygame.K_a]:
        dtheta += STEP_RADIANS_PER_TICK
    if pressed_keys[pygame.K_d]:
        dtheta -= STEP_RADIANS_PER_TICK
    if pressed_keys[pygame.K_UP]:
        dphi -= STEP_RADIANS_PER_TICK
    if pressed_keys[pygame.K_DOWN]:
        dphi += STEP_RADIANS_PER_TICK
    return dr, dtheta, dphi


def save_history(history: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]) -> None:
    """Prompt whether to save recorded (state, action) pairs, and if so, write them
    as two datetime-stamped .npy files (states_*.npy, actions_*.npy) -- states[i]
    is the state that was current when actions[i] was chosen and sent."""
    if not history:
        print("Nothing recorded, skipping save.")
        return

    answer = input(f"Save {len(history)} recorded state/action pairs? [y/N]: ").strip().lower()
    if answer != "y":
        print("Discarded.")
        return

    directory_input = input(f"Save directory [{DEFAULT_SAVE_DIRECTORY}]: ").strip()
    save_directory = Path(directory_input) if directory_input else DEFAULT_SAVE_DIRECTORY
    save_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    states_path = save_directory / f"states_{timestamp}.npy"
    actions_path = save_directory / f"actions_{timestamp}.npy"
    np.save(states_path, np.stack([state for state, _ in history]))
    np.save(actions_path, np.stack([action for _, action in history]))
    print(f"Saved {len(history)} pairs to {states_path} and {actions_path}")


def main() -> None:
    print("Loading IK chain...")
    trimmed_urdf_path = ik.trim_urdf_subtree(str(URDF_PATH), GRIPPER_ROOT_LINK)
    chain = ik.load_arm_chain(trimmed_urdf_path, num_active_joints=NUM_POSITION_JOINTS)
    bounds = ik.joint_bounds(chain)

    print(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = rc.connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    print("Connected.")

    pygame.init()
    pygame.display.set_mode((480, 160))
    pygame.display.set_caption("Teleop controls: WASD + Up/Down move, Space toggles gripper")
    clock = pygame.time.Clock()

    r, theta, phi = cartesian_to_spherical(HOME_POSITION_FLU)
    previous_angles = np.concatenate([np.zeros(NUM_POSITION_JOINTS), WRIST_JOINT_ANGLES])
    gripper_open = False  # target state, toggled by Space
    gripper_command = GRIPPER_CLOSED_COMMAND  # current command, ramped towards the target each tick

    # (state, action) pairs, where state is whatever was current when action was
    # chosen and sent -- see the first iteration's handling below for how state
    # gets bootstrapped, since Unity only ever reports a state in response to
    # having already received an action.
    history: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    state: npt.NDArray[np.float64] | None = None
    previous_action: npt.NDArray[np.float64] | None = None
    total_ticks = 0

    print(__doc__)
    try:
        with sock:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        gripper_open = not gripper_open

                pressed_keys = pygame.key.get_pressed()
                dr, dtheta, dphi = compute_spherical_delta(pressed_keys)
                r = np.clip(r + dr, R_MIN_METERS, R_MAX_METERS)
                theta = theta + dtheta
                phi = np.clip(phi + dphi, PHI_MIN_RADIANS, PHI_MAX_RADIANS)
                target_position_flu = spherical_to_cartesian(r, theta, phi)

                angles = ik.solve_ik(chain, target_position_flu, initial_angles=previous_angles)
                normalized_arm_actions = ik.angles_to_normalized(angles, bounds)
                gripper_target_command = GRIPPER_OPEN_COMMAND if gripper_open else GRIPPER_CLOSED_COMMAND
                gripper_command = step_toward(gripper_command, gripper_target_command, GRIPPER_STEP_PER_TICK)
                action = np.concatenate([normalized_arm_actions, [gripper_command]])
                action_is_nonzero = previous_action is None or not np.allclose(action, previous_action)

                if state is not None:
                    # state was current *before* this tick's action was decided;
                    # the very first action has no such prior state to pair with.
                    total_ticks += 1
                    if action_is_nonzero or not RECORD_ONLY_NONZERO_ACTIONS:
                        history.append((state, action))
                previous_action = action

                rc.send_action(sock, normalized_arm_actions, gripper_command)
                observation = rc.read_observation(sock)
                state = rc.observation_to_array(observation)
                previous_angles = angles

                eef_to_target = np.linalg.norm(
                    observation.end_effector_position_ruf - observation.block_position_ruf
                )
                target_to_placement = np.linalg.norm(observation.block_position_ruf - observation.goal_position_ruf)
                print(
                    f"eef-target: {eef_to_target:.3f}m  target-placement: {target_to_placement:.3f}m",
                    end="\r",
                )
                clock.tick(60)
    except (KeyboardInterrupt, OSError) as e:
        print(f"\nStopping ({e}).")
    finally:
        pygame.quit()

    if RECORD_ONLY_NONZERO_ACTIONS:
        print(f"Recorded {len(history)} of {total_ticks} ticks (unchanged-action ticks skipped).")
    save_history(history)


if __name__ == "__main__":
    main()
