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
PHI_MAX_RADIANS = np.radians(80.0)

STEP_METERS_PER_TICK = 0.005
STEP_RADIANS_PER_TICK = np.radians(1.0)
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSED_COMMAND = -1.0


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
    gripper_open = False

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
                gripper_command = GRIPPER_OPEN_COMMAND if gripper_open else GRIPPER_CLOSED_COMMAND

                rc.send_action(sock, normalized_arm_actions, gripper_command)
                observation = rc.read_observation(sock)
                previous_angles = angles

                print(
                    f"r={r:.3f} theta={np.degrees(theta):.1f}deg phi={np.degrees(phi):.1f}deg "
                    f"target(FLU)={target_position_flu.round(3)} "
                    f"eef(RUF)={observation.end_effector_ruf.round(3)} "
                    f"gripper={'open' if gripper_open else 'closed'}",
                    end="\r",
                )
                clock.tick(60)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
