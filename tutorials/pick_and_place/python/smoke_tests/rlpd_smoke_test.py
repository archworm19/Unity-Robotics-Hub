"""Smoke test for RLPDPolicy: a toy double-integrator regulation task --
keep a 1D object at position 0, action applies acceleration.

  state:  [position, velocity]                      (2D)
  action: acceleration, clipped to [-1, 1]           (1D)
  reward: -abs(position) each step
  reset:  after 100 steps -- a pure time-limit truncation, not a true
          terminal state, so done is always False (treating it as a
          termination would bias the learned value function -- Pardo et al.
          2018, "Time Limits in Reinforcement Learning")

This exercises the actual SAC/RLPD gradient machinery end-to-end against real
(if trivial) dynamics -- unlike tests/test_rlpd_policy.py, which only
validates it against synthetic data. Success here means episode return should
trend towards 0 (the agent learns to hold position near 0) over training.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available when run headless/via a tool call
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rlpd_policy import RLPDPolicy

DT = 0.1
VELOCITY_DAMPING = 0.98  # mild friction -- keeps early, undertrained rollouts numerically bounded
MAX_EPISODE_STEPS = 100

NUM_ENV_STEPS = 20_000
PRINT_EVERY_EPISODES = 20
HIDDEN_SIZE = 64

NUM_EVAL_RUNS = 2
PLOT_PATH = Path(__file__).parent / "rlpd_position_vs_time.png"


class DoubleIntegratorEnv:
    """1D double integrator: action is acceleration, applied to velocity,
    which is applied to position. state = [position, velocity]."""

    def __init__(
        self,
        dt: float = DT,
        velocity_damping: float = VELOCITY_DAMPING,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        self.dt = dt
        self.velocity_damping = velocity_damping
        self.max_episode_steps = max_episode_steps
        self._position = 0.0
        self._velocity = 0.0
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._position = np.random.uniform(-1.0, 1.0)
        self._velocity = np.random.uniform(-1.0, 1.0)
        self._step_count = 0
        return self._state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Returns (post_state, reward, truncated). truncated is a pure time
        limit -- callers should treat it as an environment reset, not a
        termination (see module docstring)."""
        acceleration = float(np.clip(action, -1.0, 1.0).reshape(-1)[0])
        self._velocity = self._velocity * self.velocity_damping + acceleration * self.dt
        self._position = self._position + self._velocity * self.dt
        reward = -abs(self._position)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        return self._state(), reward, truncated

    def _state(self) -> np.ndarray:
        return np.array([self._position, self._velocity], dtype=np.float32)


def rollout_positions(env: DoubleIntegratorEnv, policy: RLPDPolicy) -> list[float]:
    """Run one episode with the (still-stochastic) trained policy and return
    the position at every step, starting with the initial reset position."""
    state = env.reset()
    positions = [float(state[0])]
    for _ in range(env.max_episode_steps):
        action = policy.act(state)
        state, _, truncated = env.step(action)
        positions.append(float(state[0]))
        if truncated:
            break
    return positions


def plot_position_vs_time(env: DoubleIntegratorEnv, policy: RLPDPolicy) -> None:
    fig, ax = plt.subplots()
    for run in range(NUM_EVAL_RUNS):
        positions = rollout_positions(env, policy)
        time = np.arange(len(positions)) * env.dt
        ax.plot(time, positions, label=f"run {run + 1}")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position")
    ax.set_title("Trained RLPDPolicy: position vs. time")
    ax.legend()
    fig.savefig(PLOT_PATH)
    print(f"\nSaved position-vs-time plot to {PLOT_PATH}")


def main() -> None:
    env = DoubleIntegratorEnv()
    policy = RLPDPolicy(state_dim=2, action_dim=1, hidden_size=HIDDEN_SIZE)

    pre_state = env.reset()
    episode_return = 0.0
    episode_returns = []

    for step in range(NUM_ENV_STEPS):
        action = policy.act(pre_state)
        post_state, reward, truncated = env.step(action)
        policy.store_transition(pre_state, action, reward, post_state, done=False)
        episode_return += reward

        if truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            pre_state = env.reset()

            if len(episode_returns) % PRINT_EVERY_EPISODES == 0:
                recent = episode_returns[-PRINT_EVERY_EPISODES:]
                print(
                    f"step {step + 1:6d}  episode {len(episode_returns):4d}  "
                    f"avg return (last {PRINT_EVERY_EPISODES}): {np.mean(recent):7.2f}"
                )
        else:
            pre_state = post_state

    first = np.mean(episode_returns[:PRINT_EVERY_EPISODES])
    last = np.mean(episode_returns[-PRINT_EVERY_EPISODES:])
    print(f"\nfirst {PRINT_EVERY_EPISODES}-episode avg return: {first:.2f}")
    print(f"last {PRINT_EVERY_EPISODES}-episode avg return:  {last:.2f}")

    plot_position_vs_time(env, policy)


if __name__ == "__main__":
    main()
