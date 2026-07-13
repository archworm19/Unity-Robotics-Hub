"""Human-in-the-loop RLPD training on Gymnasium's LunarLanderContinuous-v3,
reusing the same control/reward machinery as rlpd_controller_v2.py -- meant to
validate that machinery (RLPDPolicy, the toggle-based intervention/handback
system, dense reward shaping) against a fast, well-understood benchmark,
decoupled from Unity's own complexity (physics timing, socket IO, IK solve
failures).

Loop, once per TICK_INTERVAL_SECONDS (same structure as rlpd_controller_v2.py):
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
     true terminal state, and treating it as one would bias the learned value
     function (Pardo et al. 2018, "Time Limits in Reinforcement Learning" --
     see also rlpd_smoke_test.py, which hit this same issue)
  5. decide the *next* action -- from the human's held keys
     (human_action_from_keys) if human_control_state is on, otherwise from
     the policy -- and loop; if the episode ended (terminated OR truncated),
     reset the environment first

Human intervention: press SPACE to toggle control (mirrors "t" in
rlpd_controller_v2.py -- arrow keys are needed for thrust here, so the toggle
moves to SPACE instead). While in control: hold Up/Down for the main engine
(Up = full throttle, Down = fully off -- LunarLanderContinuous's main engine
only fires for positive values, so "off" and "reverse" are the same thing
here) and Left/Right for the lateral thrusters.

Unlike rlpd_controller_v2.py, there's no IK/pose-delta conversion layer here:
LunarLanderContinuous's own action space (2-dim, [-1, 1]: main engine,
lateral thrusters) is *already* exactly what a human naturally wants to
control, so both the human's action and the policy's action go to env.step()
directly, with nothing in between.

Reward is the environment's own (already well-shaped: distance/velocity/angle
to the pad, leg contact, fuel use, +100 landing, -100 crashing), plus
INTERVENTION_PENALTY / HANDBACK_REWARD layered on top exactly as in
rlpd_controller_v2.py.

Rendering: gymnasium owns the pygame window here (render_mode="human"), not
us. Unlike rlpd_controller_v2.py's LogWindow, a second pygame window isn't an
option (pygame.display manages exactly one), so logging here just goes to the
terminal -- reading keyboard state still works normally, since it comes from
pygame's global input state, not from whichever code happened to create the
window.
"""

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rlpd_policy import RLPDPolicy

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
DEFAULT_CHECKPOINT_PATH = TELEOP_DIRECTORY / "lunar_lander_rlpd.pt"
CHECKPOINT_EVERY_EPISODES = 10

# Larger in magnitude than a plain crash -- a human having to step in is a
# stronger negative signal than just letting the episode end on its own.
INTERVENTION_PENALTY = -10.0
# The human's last action right before handing control back to the policy --
# the state they chose to hand back from is exactly what the policy should
# learn to reach and continue from on its own.
HANDBACK_REWARD = 5.0

if abs(HANDBACK_REWARD) > abs(INTERVENTION_PENALTY):
    raise ValueError(
        f"HANDBACK_REWARD magnitude ({abs(HANDBACK_REWARD)}) must not exceed "
        f"INTERVENTION_PENALTY magnitude ({abs(INTERVENTION_PENALTY)}) -- otherwise a human "
        "handing back control could outweigh the penalty for making them intervene in the first place."
    )

# A landing/crash's terminal bonus/penalty (+/-100) dwarfs ordinary shaping
# reward (usually within a couple points per tick) -- used to tell success
# apart from a crash on a terminated step, see record_episode_end's caller.
SUCCESS_REWARD_THRESHOLD = 50.0

# How long each loop iteration waits for a human reaction, continuously
# sampling keyboard state the whole time (see wait_and_sample_input) -- also
# how much wall-clock time separates consecutive stored transitions.
TICK_INTERVAL_SECONDS = 0.025
POLL_INTERVAL_SECONDS = 0.02  # sampling granularity within that wait

AVERAGE_OVER_EPISODES = 10


def wait_and_sample_input(duration_seconds: float) -> tuple[object, bool, bool]:
    """Continuously polls pygame's event queue and key state for
    duration_seconds. Returns (pressed_keys, toggle_requested,
    quit_requested): pressed_keys is the *last* sample taken, right at the
    end of the window; toggle_requested is True if SPACE was pressed
    (KEYDOWN) at any point during the window (edge-triggered -- control
    toggles once per press, not once per tick it happens to still be held)."""
    deadline = time.monotonic() + duration_seconds
    toggle_requested = False
    quit_requested = False
    pressed_keys = pygame.key.get_pressed()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                toggle_requested = True
        pressed_keys = pygame.key.get_pressed()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))
    return pressed_keys, toggle_requested, quit_requested


def human_action_from_keys(pressed_keys) -> np.ndarray:
    """2-dim (main engine, lateral thrusters) action in the exact same
    [-1, 1]-per-dimension space the policy outputs -- +1/-1 for whichever key
    (if any) is held for that dimension, 0 otherwise. Up = full main engine,
    Down = fully off (see module docstring for why those aren't opposites
    here). Right = right thruster, Left = left thruster."""
    main = 0.0
    if pressed_keys[pygame.K_UP]:
        main += 1.0
    if pressed_keys[pygame.K_DOWN]:
        main -= 1.0
    lateral = 0.0
    if pressed_keys[pygame.K_RIGHT]:
        lateral += 1.0
    if pressed_keys[pygame.K_LEFT]:
        lateral -= 1.0
    return np.array([main, lateral], dtype=np.float32)


def main() -> None:
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    pygame.init()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = RLPDPolicy(state_dim=state_dim, action_dim=action_dim)
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
    env.render()
    human_control_state = False
    action = policy.act(state)
    tick = 0

    print(__doc__)
    try:
        while True:
            clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
            next_state, raw_reward, terminated, truncated, _ = env.step(clipped_action)
            env.render()

            pressed_keys, toggle_requested, quit_requested = wait_and_sample_input(TICK_INTERVAL_SECONDS)
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

            human_action = human_action_from_keys(pressed_keys)
            q_estimate = policy.estimate_q(state, action)

            # terminated, not "terminated or truncated" -- see module docstring.
            policy.store_transition(
                state, action, reward, next_state, terminated,
                override_action=action if human_control_state else None,
            )
            episode_return += reward

            print(
                f"tick={tick:6d} source={'human' if human_control_state else 'policy':6s} "
                f"Q={q_estimate:+7.3f} reward={reward:+.2f}"
            )
            tick += 1

            episode_over = terminated or truncated
            if episode_over:
                record_episode_end(success=terminated and raw_reward > SUCCESS_REWARD_THRESHOLD)
                next_state, _ = env.reset()
                env.render()
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
