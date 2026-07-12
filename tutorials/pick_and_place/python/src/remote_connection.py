"""Shared TCP client for RemoteJointController.cs's wire protocol.

Wire format, one exchange per simulation tick, all values little-endian float32:

  Python -> Unity (8 floats): 6 arm-joint targets in [-1, 1], then 1 gripper
                   target in [-1, 1] (-1 = fully closed, +1 = fully open), then
                   1 reset flag (nonzero = reset both the block and the arm to
                   their spawn/home poses before applying this tick's targets;
                   see send_action)

  Unity -> Python (25 floats), all physical/sensed state (not commanded
                   values), positions and rotations relative to base_link, in
                   Unity's RUF frame:
    [0:3]   end-effector position (x, y, z)
    [3:7]   end-effector rotation (x, y, z, w quaternion)
    [7:8]   gripper width (meters, actual distance between the fingers)
    [8:14]  arm joint positions (radians, actual -- joint_1 .. joint_6)
    [14:17] goal (TargetPlacement) position (x, y, z)
    [17:20] block ("Target") position (x, y, z)
    [20:24] block rotation (x, y, z, w quaternion)
    [24:25] placement state (float-encoded PlacementState)

Episodes reset automatically on the Unity side: once placement_state comes back
as INSIDE_PLACED (success) or FAILED_FELL (the block fell off the table), that
observation reflects the terminal state as usual, and Unity then resets the
block (not the arm) before processing the next action -- so the very next
observation already reflects a fresh episode, with no ticks or actions lost to
the reset itself. That automatic reset never touches the arm; send_action's
reset flag is the only way to reset it too (see its docstring for why a caller
would want that).

Unity is fully agnostic to user input -- it only ever executes this 8-float
signal. Physics only advances one tick per action received (RemoteJointController
is in lockstep), so the tick rate is driven entirely by how fast this client sends.
"""

import socket
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

NUM_ARM_JOINTS = 6
NUM_POLICY_ACTION_FLOATS = NUM_ARM_JOINTS + 1  # + gripper -- what a Policy actually outputs (see policy.py)
NUM_ACTION_FLOATS = NUM_POLICY_ACTION_FLOATS + 1  # + reset flag -- the full wire size (see send_action)
NUM_OBSERVATION_FLOATS = 25


class PlacementState(IntEnum):
    """Mirrors RemoteJointController's WirePlacementState exactly -- whether the
    block is outside the goal zone, inside but still moving, inside and settled
    (Rigidbody velocity below TargetPlacement's threshold), or fell off the
    table. INSIDE_PLACED and FAILED_FELL are both episode-terminal: Unity resets
    the block automatically once either is reached (see module docstring)."""

    OUTSIDE = 0
    INSIDE_FLOATING = 1
    INSIDE_PLACED = 2
    FAILED_FELL = 3


@dataclass(frozen=True)
class Observation:
    end_effector_position_ruf: npt.NDArray[np.float64]
    end_effector_orientation_ruf: npt.NDArray[np.float64]  # quaternion (x, y, z, w)
    gripper_width_meters: float
    joint_positions_radians: npt.NDArray[np.float64]  # 6 actual arm joint angles
    goal_position_ruf: npt.NDArray[np.float64]
    block_position_ruf: npt.NDArray[np.float64]
    block_orientation_ruf: npt.NDArray[np.float64]  # quaternion (x, y, z, w)
    placement_state: PlacementState


def connect_with_retry(host: str, port: int, timeout: float) -> socket.socket:
    """Connect to (host, port), retrying every 0.5s until it accepts or timeout elapses.

    Unity is the TCP server here (RemoteJointController.Start() opens the listener
    only once Play begins), so a bare connect attempt made before that would just
    fail -- this covers the ordinary "script started before Play was pressed" case.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
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


def send_action(
    sock: socket.socket,
    normalized_arm_actions: npt.ArrayLike,
    gripper_command: float,
    reset: bool = False,
) -> None:
    """Send one (arm joint targets, gripper target, reset flag) action to
    RemoteJointController, each joint/gripper value in [-1, 1].

    reset=True asks Unity to reset both the block and the arm to their
    spawn/home poses before applying this tick's targets -- unlike the
    automatic, success/fall-triggered reset (see module docstring), which
    never touches the arm. Meant for a caller-detected unrecoverable
    situation the arm itself got into (e.g. rlpd_controller.py resetting after
    an IK solve failure), not for ordinary episode boundaries.
    """
    values = list(normalized_arm_actions) + [gripper_command, 1.0 if reset else 0.0]
    sock.sendall(struct.pack(f"<{NUM_ACTION_FLOATS}f", *values))


def read_observation(sock: socket.socket) -> Observation:
    """Read one observation (see module docstring for the field layout) from
    RemoteJointController."""
    raw = recv_exact(sock, NUM_OBSERVATION_FLOATS * 4)
    v = struct.unpack(f"<{NUM_OBSERVATION_FLOATS}f", raw)
    return Observation(
        end_effector_position_ruf=np.array(v[0:3]),
        end_effector_orientation_ruf=np.array(v[3:7]),
        gripper_width_meters=v[7],
        joint_positions_radians=np.array(v[8:14]),
        goal_position_ruf=np.array(v[14:17]),
        block_position_ruf=np.array(v[17:20]),
        block_orientation_ruf=np.array(v[20:24]),
        placement_state=PlacementState(round(v[24])),
    )


def observation_to_array(observation: Observation) -> npt.NDArray[np.float64]:
    """Flatten an Observation back into the same 25-float layout read_observation
    parses it from -- e.g. for logging/recording a raw, self-documented array."""
    return np.concatenate([
        observation.end_effector_position_ruf,
        observation.end_effector_orientation_ruf,
        [observation.gripper_width_meters],
        observation.joint_positions_radians,
        observation.goal_position_ruf,
        observation.block_position_ruf,
        observation.block_orientation_ruf,
        [float(observation.placement_state)],
    ])
