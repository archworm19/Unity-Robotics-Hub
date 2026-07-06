"""Drive Unity through the Policy interface (see policy.py) with either
Replay (replaying a previously recorded teleop trajectory, to sanity-check
that a recording actually reproduces a sensible run) or a trained
BehaviorCloningMLP -- pick with --policy {replay,mlp}.

Replay plays back the recorded actions.npy (with a few repeats per action --
see Replay.pause_timesteps) rather than computing anything from the observed
state, and reports a live drift against its own recorded ground truth (see
Replay.drift) if a matching states_*.npy was found alongside it. A trained
policy has no such ground-truth trajectory to compare against, so it just
runs, state in/action out, until interrupted.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import remote_connection as rc
from policy import Replay
from train_bc_mlp import BehaviorCloningMLP, MODEL_SAVE_PATH

HOST = "127.0.0.1"
PORT = 9000
CONNECT_TIMEOUT_SECONDS = 30.0

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
ACTIONS_PATH = TELEOP_DIRECTORY / "actions_20260706_143202.npy"

DEFAULT_REPLAY_ACTIONS_PATH = ACTIONS_PATH
DEFAULT_MLP_CHECKPOINT_PATH = MODEL_SAVE_PATH


def build_replay_policy(actions_path: Path = DEFAULT_REPLAY_ACTIONS_PATH) -> Replay:
    """Build a Replay policy from a recorded actions.npy file."""
    return Replay.load_from_checkpoint(data_location=actions_path, checkpoint_location=Path())


def build_mlp_policy(checkpoint_path: Path = DEFAULT_MLP_CHECKPOINT_PATH) -> BehaviorCloningMLP:
    """Build a trained behavior-cloning MLP policy from a saved checkpoint."""
    return BehaviorCloningMLP.load_from_checkpoint(data_location=Path(), checkpoint_location=checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["replay", "mlp"],
        default="replay",
        help="Which policy to build and run (default: replay).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = build_replay_policy() if args.policy == "replay" else build_mlp_policy()
    print(f"Using {args.policy} policy.")

    print(f"Connecting to Unity at {HOST}:{PORT}... make sure you've pressed Play in the Editor.")
    sock = rc.connect_with_retry(HOST, PORT, CONNECT_TIMEOUT_SECONDS)
    print("Connected.")

    state = np.zeros(rc.NUM_OBSERVATION_FLOATS)  # placeholder; overwritten after the first exchange
    try:
        with sock:
            while not (isinstance(policy, Replay) and policy.finished):
                action = policy.act(state)
                rc.send_action(sock, action[: rc.NUM_ARM_JOINTS], action[rc.NUM_ARM_JOINTS])
                observation = rc.read_observation(sock)
                state = rc.observation_to_array(observation)

                status = f"live_eef={observation.end_effector_position_ruf.round(3)}"
                if hasattr(policy, "drift"):
                    drift = policy.drift(state)
                    if drift is not None:
                        status += f" drift={drift:.3f}m"
                print(status, end="\r")
    except (KeyboardInterrupt, OSError) as e:
        print(f"\nStopping ({e}).")
        return
    print("\nDone.")


if __name__ == "__main__":
    main()
