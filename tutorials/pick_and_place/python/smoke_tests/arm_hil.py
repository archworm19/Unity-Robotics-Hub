"""Human-in-the-loop RLPD training on ArmEnv (see arm_env.py), reusing the
same control/reward machinery as lunar_lander_hil.py / rlpd_controller_v2.py.

Loop, once per TICK_INTERVAL_SECONDS (same structure as the other two):
  1. step the environment with the action decided last iteration; read back
     the resulting state and its native reward
  2. wait TICK_INTERVAL_SECONDS, continuously sampling keyboard state the
     whole time (see wait_and_sample_input) -- the human gets this whole
     window to react, and to press SPACE if they want to toggle control
  3. if SPACE was pressed, flip human_control_state. If that flip just handed
     control *to* a human, add INTERVENTION_PENALTY to the reward; if it just
     handed control *back to* the policy, add HANDBACK_REWARD instead
  4. store the (state, action, reward, next_state, terminated) transition --
     terminated, not "terminated or truncated": a time-limit cutoff isn't a
     true terminal state (Pardo et al. 2018 -- see lunar_lander_hil.py and
     rlpd_smoke_test.py, which hit this same issue)
  5. decide the *next* action -- from the human's key duty cycles
     (human_action_from_duty_cycles) if human_control_state is on, otherwise
     from the policy -- and loop; if the episode ended (terminated OR truncated),
     reset the environment first

Human intervention: press SPACE to toggle control. While in control:
  W/S = forward/backward              (base translation, relative to wherever
                                        the gripper is currently facing, not a
                                        fixed screen direction; see arm_env.py's
                                        step() -- no lateral/strafe control,
                                        translation is forward/backward only)
  A/D = counterclockwise/clockwise    (base rotation)
  O/P = +gripper/-gripper             (open/close -- same mapping as
                                        rlpd_controller_v2.py's O/P)

Reward is ArmEnv's own (already ported wholesale from rlpd_controller_v2.py's
structure -- see arm_env.py), plus INTERVENTION_PENALTY / HANDBACK_REWARD
layered on top exactly as in the other two HIL scripts.

Human actions are a *duty cycle*, not a single keyboard snapshot: each of the
8 movement/gripper keys' held-fraction over the TICK_INTERVAL_SECONDS window
is tracked continuously (see wait_and_sample_input), not just sampled once at
the end of it. A single end-of-window sample can only ever produce exactly
{-1, 0, +1} per action dimension, while the policy's own actions are
continuous samples from a Gaussian -- meaning every human-sourced transition
in the buffer would sit exactly on the corners/edges of the action space, and
SAC's policy update explicitly hill-climbs Q(s, pi(s)) over the *continuous*
interior of that space. A critic trained almost entirely on those corners has
no grounding for the interior points the policy is being pushed toward, which
is exactly the setup for it to get pulled toward spurious, untrained-on
Q-value overestimates rather than genuinely good actions (the same
extrapolation-error problem offline-RL methods like CQL/BCQ exist to guard
against). Duty cycle -- how much of the window a key was actually held --
gives human actions real intermediate values too, so the buffer's human
actions actually cover the space the policy is being optimized over, not just
its corners.

Rendering: ArmEnv owns the pygame window here (render_mode="human"), same
reasoning as lunar_lander_hil.py -- logging goes to the terminal, not a
second pygame window.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arm_env import ArmEnv, OBSERVATION_DIM, ACTION_DIM
from rlpd_policy import RLPDPolicy

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
DEFAULT_CHECKPOINT_PATH = TELEOP_DIRECTORY / "arm_rlpd.pt"
CHECKPOINT_EVERY_EPISODES = 10

# Larger in magnitude than a plain failure -- a human having to step in is a
# stronger negative signal than just letting the episode end on its own.
INTERVENTION_PENALTY = 0.0
# The human's last action right before handing control back to the policy --
# the state they chose to hand back from is exactly what the policy should
# learn to reach and continue from on its own.
HANDBACK_REWARD = 0.0

if abs(HANDBACK_REWARD) > abs(INTERVENTION_PENALTY):
    raise ValueError(
        f"HANDBACK_REWARD magnitude ({abs(HANDBACK_REWARD)}) must not exceed "
        f"INTERVENTION_PENALTY magnitude ({abs(INTERVENTION_PENALTY)}) -- otherwise a human "
        "handing back control could outweigh the penalty for making them intervene in the first place."
    )

# How long each loop iteration waits for a human reaction, continuously
# sampling keyboard state the whole time (see wait_and_sample_input) -- also
# how much wall-clock time separates consecutive stored transitions. Matches
# ArmEnv's own default dt, so one HIL tick == one env.step() worth of sim time.
TICK_INTERVAL_SECONDS = 0.2
POLL_INTERVAL_SECONDS = 0.02  # sampling granularity within that wait

AVERAGE_OVER_EPISODES = 10

# The 6 movement/gripper keys duty-cycle tracking covers -- see
# wait_and_sample_input / human_action_from_duty_cycles.
TRACKED_KEYS = (
    pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_o, pygame.K_p,
)


def wait_and_sample_input(duration_seconds: float) -> tuple[dict, bool, bool]:
    """Continuously polls pygame's event queue and key state for
    duration_seconds. Returns (key_duty_cycles, toggle_requested,
    quit_requested): key_duty_cycles maps each of TRACKED_KEYS to the
    *fraction* of the window it was held down (sampled at POLL_INTERVAL_SECONDS
    granularity and averaged), not just whether it happened to be held at one
    instant -- see module docstring for why that distinction matters.
    toggle_requested is True if SPACE was pressed (KEYDOWN) at any point
    during the window (edge-triggered -- control toggles once per press, not
    once per tick it happens to still be held)."""
    deadline = time.monotonic() + duration_seconds
    toggle_requested = False
    quit_requested = False
    held_counts = dict.fromkeys(TRACKED_KEYS, 0)
    num_samples = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                toggle_requested = True
        pressed_keys = pygame.key.get_pressed()
        for key in TRACKED_KEYS:
            if pressed_keys[key]:
                held_counts[key] += 1
        num_samples += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))
    key_duty_cycles = {key: held_counts[key] / num_samples for key in TRACKED_KEYS}
    return key_duty_cycles, toggle_requested, quit_requested


def human_action_from_duty_cycles(key_duty_cycles: dict) -> np.ndarray:
    """3-dim (forward, angular_accel, gripper) action in the exact same
    [-1, 1]-per-dimension space the policy outputs -- each dimension is the
    difference of its two keys' duty cycles (continuous in [-1, 1], not just
    {-1, 0, +1}; see module docstring). forward is relative to the gripper's
    own current facing direction, not a fixed screen direction -- see
    arm_env.py's step(). A = counterclockwise (+angular), D = clockwise
    (-angular)."""
    forward = key_duty_cycles[pygame.K_w] - key_duty_cycles[pygame.K_s]
    angular = key_duty_cycles[pygame.K_a] - key_duty_cycles[pygame.K_d]
    gripper = key_duty_cycles[pygame.K_o] - key_duty_cycles[pygame.K_p]
    return np.array([forward, angular, gripper], dtype=np.float32)


def main() -> None:
    env = ArmEnv(render_mode="human", dt=TICK_INTERVAL_SECONDS)
    pygame.init()

    policy = RLPDPolicy(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)
    DEFAULT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    episode_return = 0.0
    episode_returns: list[float] = []
    num_successes = 0

    def record_episode_end(success: bool) -> None:
        nonlocal episode_return, num_successes
        episode_returns.append(episode_return)
        if success:
            num_successes += 1
        episode_return = 0.0
        recent = episode_returns[-AVERAGE_OVER_EPISODES:]
        print(
            f"episode {len(episode_returns):4d}  successes: {num_successes}/{len(episode_returns)}  "
            f"avg return (last {len(recent)}): {np.mean(recent):+7.2f}"
        )
        if len(episode_returns) % CHECKPOINT_EVERY_EPISODES == 0:
            policy.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
            print(f"Saved checkpoint to {DEFAULT_CHECKPOINT_PATH}")

    state, _ = env.reset()
    human_control_state = False
    action = policy.act(state)
    tick = 0

    print(__doc__)
    try:
        while True:
            clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
            next_state, raw_reward, terminated, truncated, info = env.step(clipped_action)

            key_duty_cycles, toggle_requested, quit_requested = wait_and_sample_input(TICK_INTERVAL_SECONDS)
            if quit_requested:
                break

            was_human_control = human_control_state
            if toggle_requested:
                human_control_state = not human_control_state

            if human_control_state and not was_human_control:
                reward = raw_reward + INTERVENTION_PENALTY
                print(f"  -> human took control (reward {raw_reward:+.2f} -> {reward:+.2f})")
            elif was_human_control and not human_control_state:
                reward = raw_reward + HANDBACK_REWARD
                print(f"  -> handing back to policy (reward {raw_reward:+.2f} -> {reward:+.2f})")
            else:
                reward = raw_reward

            human_action = human_action_from_duty_cycles(key_duty_cycles)
            q_estimate = policy.estimate_q(state, action)

            # terminated, not "terminated or truncated" -- see module docstring.
            policy.store_transition(
                state, action, reward, next_state, terminated,
                override_action=action if human_control_state else None,
            )
            episode_return += reward

            # Qpi/Qreal/alpha reflect the most recent training batch (from the
            # store_transition() call just above) -- see RLPDPolicy's
            # last_policy_action_q_mean/last_replay_action_q_mean/alpha for
            # what they're diagnosing (roughly: is the policy actually
            # improving relative to what's in the buffer, and is entropy
            # tuning keeping it noisier than useful).
            qpi = policy.last_policy_action_q_mean
            qreal = policy.last_replay_action_q_mean
            qpi_str = f"{qpi:+7.3f}" if qpi is not None else "   n/a "
            qreal_str = f"{qreal:+7.3f}" if qreal is not None else "   n/a "
            print(
                f"tick={tick:6d} source={'human' if human_control_state else 'policy':6s} "
                f"gripped={env.is_gripped!s:5} Q={q_estimate:+7.3f} "
                f"Qpi={qpi_str} Qreal={qreal_str} alpha={policy.alpha:.3f} reward={reward:+.2f}"
            )
            tick += 1

            episode_over = terminated or truncated
            if episode_over:
                record_episode_end(success=bool(info.get("success", False)))
                next_state, _ = env.reset()
                human_control_state = False

            state = next_state
            action = human_action if human_control_state else policy.act(state)
    except KeyboardInterrupt as e:
        print(f"Stopping ({e}).")
    finally:
        env.close()

    policy.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
    print(f"Saved final checkpoint to {DEFAULT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
