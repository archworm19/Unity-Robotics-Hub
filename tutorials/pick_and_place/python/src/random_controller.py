"""Experiment 3: sample a random end-effector pose, solve IK for it, and drive
the arm there over the RemoteJointController socket.

Loop:
  1. Sample a random end-effector pose (position + small orientation
     perturbation) within the arm's reachable workspace.
  2. Solve IK to get the 6 joint angles that reach it.
  3. Send the corresponding normalized action to Unity.
  4. Physics only advances one tick per message received (RemoteJointController
     is in lockstep), so "waiting" for the arm to get there means resending
     the same action repeatedly until the reported end-effector position is
     within tolerance of the target, or a max number of ticks is reached.
  5. Print the result and repeat.

Positions are exchanged with Unity in RUF (X-right, Y-up, Z-forward),
relative to base_link. ikpy solves in the URDF's native FLU (X-forward,
Y-left, Z-up) frame, so every target/observation is converted at the
boundary -- see ik.flu_to_ruf / ik.ruf_to_flu.
"""

import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent))
import ik

HOST = "127.0.0.1"
PORT = 9000
NUM_JOINTS = 6
NUM_OBSERVATION_FLOATS = 6  # end-effector xyz + object xyz
CONNECT_TIMEOUT_SECONDS = 30.0

URDF_PATH = Path(__file__).parent / ".." / ".." / "PickAndPlaceProject" / "Assets" / "URDF" / "niryo_one" / "niryo_one.urdf"

# niryo_one.urdf specific: the gripper mechanism isn't part of the arm's own
# 6-DOF chain, and one of its joints has a malformed axis that crashes ikpy's
# parser, so it's pruned before handing the URDF to ikpy.
GRIPPER_ROOT_LINK = "gripper_base"

# Reachable workspace, FLU frame (X-forward, Y-left, Z-up), relative to base_link.
POSITION_LOW_FLU = np.array([0.15, -0.2, 0.15])
POSITION_HIGH_FLU = np.array([0.35, 0.2, 0.45])
MAX_ORIENTATION_PERTURBATION_DEGREES = 20.0

POSITION_TOLERANCE_METERS = 0.01
MAX_HOLD_STEPS = 200


def connect_with_retry(host: str, port: int, timeout: float) -> socket.socket:
    """Connect to (host, port), retrying every 0.5s until it accepts or timeout elapses.

    Unity is the TCP server here (RemoteJointController.Start() opens the listener
    only once Play begins), so a bare connect attempt made before that would just
    fail -- this covers the ordinary "script started before Play was pressed" case.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, port))
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Read exactly num_bytes from sock, looping over partial reads."""
    buffer = bytearray()
    while len(buffer) < num_bytes:
        chunk = sock.recv(num_bytes - len(buffer))
        if not chunk:
            raise ConnectionError("Socket closed while reading.")
        buffer.extend(chunk)
    return bytes(buffer)


def send_action(sock: socket.socket, normalized_actions: npt.ArrayLike) -> None:
    """Send one joint-target action ([-1, 1] per joint) to RemoteJointController."""
    sock.sendall(struct.pack(f"<{NUM_JOINTS}f", *normalized_actions))


def read_observation(sock: socket.socket) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read one observation from RemoteJointController: (end_effector_ruf, object_ruf),
    each relative to base_link, in Unity's RUF frame."""
    raw = recv_exact(sock, NUM_OBSERVATION_FLOATS * 4)
    values = struct.unpack(f"<{NUM_OBSERVATION_FLOATS}f", raw)
    return np.array(values[0:3]), np.array(values[3:6])


def random_small_rotation(max_degrees: float) -> npt.NDArray[np.float64]:
    """A random rotation matrix about a uniformly random axis, with angle magnitude
    up to max_degrees (Rodrigues' rotation formula)."""
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(np.random.uniform(-max_degrees, max_degrees))
    kx, ky, kz = axis
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def sample_target_pose(
    home_orientation_flu: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample a random (position, orientation) target pose, FLU frame, within the
    arm's reachable workspace: position uniform in a bounding box, orientation a
    small random perturbation of home_orientation_flu (full random SO(3) samples
    would mostly be unreachable given this arm's joint limits)."""
    position_flu = np.random.uniform(POSITION_LOW_FLU, POSITION_HIGH_FLU)
    orientation_flu = random_small_rotation(MAX_ORIENTATION_PERTURBATION_DEGREES) @ home_orientation_flu
    return position_flu, orientation_flu


def drive_to_target(
    sock: socket.socket, normalized_actions: npt.ArrayLike, target_position_flu: npt.ArrayLike
) -> tuple[bool, int, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resend normalized_actions each simulation tick (RemoteJointController only
    advances physics one step per message received) until the reported end-effector
    position is within POSITION_TOLERANCE_METERS of target_position_flu, or
    MAX_HOLD_STEPS ticks elapse.

    Returns (reached, steps_taken, final_error_meters, end_effector_ruf, object_ruf).
    """
    target_position_ruf = ik.flu_to_ruf(target_position_flu)
    for step in range(MAX_HOLD_STEPS):
        send_action(sock, normalized_actions)
        end_effector_ruf, object_ruf = read_observation(sock)
        error = np.linalg.norm(end_effector_ruf - target_position_ruf)
        if error < POSITION_TOLERANCE_METERS:
            return True, step + 1, error, end_effector_ruf, object_ruf
    return False, MAX_HOLD_STEPS, error, end_effector_ruf, object_ruf


def main() -> None:
    print("Loading IK chain...")
    trimmed_urdf_path = ik.trim_urdf_subtree(str(URDF_PATH), GRIPPER_ROOT_LINK)
    chain = ik.load_arm_chain(trimmed_urdf_path)
    bounds = ik.joint_bounds(chain)
    home_angles = np.zeros(NUM_JOINTS)
    home_orientation_flu = np.eye(3)  # arm's home pose has identity orientation (see calibration check)

    print(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Connected.")

    previous_angles = home_angles
    try:
        with sock:
            while True:
                target_position_flu, target_orientation_flu = sample_target_pose(home_orientation_flu)
                angles = ik.solve_ik(
                    chain, target_position_flu, target_orientation_flu, initial_angles=previous_angles
                )
                normalized_actions = ik.angles_to_normalized(angles, bounds)

                reached, steps, error, end_effector_ruf, object_ruf = drive_to_target(
                    sock, normalized_actions, target_position_flu
                )
                previous_angles = angles

                status = "reached" if reached else "TIMED OUT"
                print(
                    f"target(FLU)={target_position_flu.round(3)} "
                    f"eef(RUF)={end_effector_ruf.round(3)} object(RUF)={object_ruf.round(3)} "
                    f"error={error:.4f}m steps={steps} [{status}]"
                )
    except KeyboardInterrupt:
        print("Stopping.")


if __name__ == "__main__":
    main()
