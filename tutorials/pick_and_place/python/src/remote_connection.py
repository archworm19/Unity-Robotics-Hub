"""Shared TCP client for RemoteJointController.cs's wire protocol.

Wire format, one exchange per simulation tick, all values little-endian float32:
  Python -> Unity: 6 arm-joint targets in [-1, 1], then 1 gripper target in
                   [-1, 1] (-1 = fully closed, +1 = fully open)
  Unity -> Python: end-effector position (x, y, z), object position (x, y, z),
                   both relative to base_link, in Unity's RUF frame

Unity is fully agnostic to user input -- it only ever executes this 7-DOF
signal. Physics only advances one tick per action received (RemoteJointController
is in lockstep), so the tick rate is driven entirely by how fast this client sends.
"""

import socket
import struct
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NUM_ARM_JOINTS = 6
NUM_ACTION_FLOATS = NUM_ARM_JOINTS + 1  # + gripper
NUM_OBSERVATION_FLOATS = 6  # end-effector xyz + object xyz


@dataclass(frozen=True)
class Observation:
    end_effector_ruf: npt.NDArray[np.float64]
    object_ruf: npt.NDArray[np.float64]


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


def send_action(sock: socket.socket, normalized_arm_actions: npt.ArrayLike, gripper_command: float) -> None:
    """Send one (arm joint targets, gripper target) action to RemoteJointController,
    each value in [-1, 1]."""
    values = list(normalized_arm_actions) + [gripper_command]
    sock.sendall(struct.pack(f"<{NUM_ACTION_FLOATS}f", *values))


def read_observation(sock: socket.socket) -> Observation:
    """Read one observation (end-effector/object position, both RUF frame relative
    to base_link) from RemoteJointController."""
    raw = recv_exact(sock, NUM_OBSERVATION_FLOATS * 4)
    values = struct.unpack(f"<{NUM_OBSERVATION_FLOATS}f", raw)
    return Observation(end_effector_ruf=np.array(values[0:3]), object_ruf=np.array(values[3:6]))
