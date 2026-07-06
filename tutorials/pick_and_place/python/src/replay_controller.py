"""Experiment 6: replay a previously recorded teleop trajectory back into the
sim, to sanity-check that a recording actually reproduces a sensible run.

Drives Unity through the same act(state) -> action interface a real policy
would eventually use (see policy.py), but backed by Replay, which just plays
back the recorded actions.npy (with a few repeats per action -- see
Replay.pause_timesteps) rather than computing anything from the observed state.
The recorded states.npy is used here only for a live drift comparison, not fed
into the replay decision at all.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import remote_connection as rc
from policy import Replay

HOST = "127.0.0.1"
PORT = 9000
CONNECT_TIMEOUT_SECONDS = 30.0

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
STATES_PATH = TELEOP_DIRECTORY / "states_20260706_143202.npy"
ACTIONS_PATH = TELEOP_DIRECTORY / "actions_20260706_143202.npy"

PAUSE_TIMESTEPS = 2


def main() -> None:
    recorded_states = np.load(STATES_PATH)
    recorded_actions = np.load(ACTIONS_PATH)
    print(f"Loaded {len(recorded_actions)} recorded (state, action) pairs from {TELEOP_DIRECTORY}")
    replay = Replay(recorded_actions, pause_timesteps=PAUSE_TIMESTEPS)

    print(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = rc.connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    print("Connected.")

    state = np.zeros(rc.NUM_OBSERVATION_FLOATS)  # Replay ignores state; a placeholder is fine to start
    try:
        with sock:
            while not replay.finished:
                action = replay.act(state)
                rc.send_action(sock, action[: rc.NUM_ARM_JOINTS], action[rc.NUM_ARM_JOINTS])
                observation = rc.read_observation(sock)
                state = rc.observation_to_array(observation)

                recorded_eef = recorded_states[replay.current_index][0:3]
                live_eef = observation.end_effector_position_ruf
                drift = np.linalg.norm(live_eef - recorded_eef)
                print(
                    f"replaying {replay.current_index + 1}/{len(recorded_actions)} "
                    f"live_eef={live_eef.round(3)} recorded_eef={recorded_eef.round(3)} drift={drift:.3f}m",
                    end="\r",
                )
    except (KeyboardInterrupt, OSError) as e:
        print(f"\nStopping ({e}).")
        return
    print("\nReplay finished.")


if __name__ == "__main__":
    main()
