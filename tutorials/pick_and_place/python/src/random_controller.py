"""Wiring test: send random joint targets to Unity over a raw TCP socket and print
back the end-effector and object positions.

Wire format, one exchange per simulation step, all values little-endian float32:
  Python -> Unity: one value per revolute joint, each in [-1, 1]
  Unity -> Python: end-effector position (x, y, z), object position (x, y, z)
"""

import socket
import struct
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 9000
NUM_JOINTS = 6
NUM_OBSERVATION_FLOATS = 6  # end-effector xyz + object xyz
CONNECT_TIMEOUT_SECONDS = 30.0


def connect_with_retry(host, port, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, port))
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def recv_exact(sock, num_bytes):
    buffer = bytearray()
    while len(buffer) < num_bytes:
        chunk = sock.recv(num_bytes - len(buffer))
        if not chunk:
            raise ConnectionError("Socket closed while reading.")
        buffer.extend(chunk)
    return bytes(buffer)


def main():
    print(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Connected.")

    try:
        with sock:
            while True:
                actions = np.random.uniform(-1.0, 1.0, size=NUM_JOINTS).astype(np.float32)
                sock.sendall(struct.pack(f"<{NUM_JOINTS}f", *actions))

                raw_observation = recv_exact(sock, NUM_OBSERVATION_FLOATS * 4)
                observation = struct.unpack(f"<{NUM_OBSERVATION_FLOATS}f", raw_observation)
                end_effector_position = observation[0:3]
                object_position = observation[3:6]
                print(f"end-effector: {end_effector_position}  object: {object_position}")
    except KeyboardInterrupt:
        print("Stopping.")


if __name__ == "__main__":
    main()
