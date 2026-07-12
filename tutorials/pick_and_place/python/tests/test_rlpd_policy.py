"""Unit tests for rlpd_policy.RLPDPolicy."""

from pathlib import Path

import numpy as np
import torch
from rlpd_policy import GaussianPolicy, RLPDPolicy
from torch import nn

STATE_DIM = 4
ACTION_DIM = 2


def test_act_shape_and_bounds():
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM, min_buffer_size=50, batch_size=16)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action = policy.act(state)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)


def test_gaussian_policy_sample_shape_and_bounds():
    policy = GaussianPolicy(STATE_DIM, ACTION_DIM)
    states = torch.randn(4, STATE_DIM)
    actions, log_probs = policy.sample(states)
    assert actions.shape == (4, ACTION_DIM)
    assert log_probs.shape == (4, 1)
    assert torch.all(actions.abs() <= 1.0)


def test_store_transition_stores_one_complete_transition_per_call():
    # huge min_buffer_size so no training happens mid-test and disturbs state
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM, min_buffer_size=10_000_000, batch_size=4)

    s0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    a0 = np.array([10.0, 10.0], dtype=np.float32)

    policy.store_transition(s0, a0, reward=1.0, post_state=s1, done=False)
    assert len(policy._online_buffer) == 1
    buf = policy._online_buffer
    assert np.array_equal(buf.states[0], s0)
    assert np.array_equal(buf.actions[0], a0)
    assert buf.rewards[0][0] == 1.0
    assert np.array_equal(buf.next_states[0], s1)
    assert buf.terminations[0][0] == 0.0

    s2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    a1 = np.array([11.0, 11.0], dtype=np.float32)
    policy.store_transition(s1, a1, reward=2.0, post_state=s2, done=True)
    assert len(policy._online_buffer) == 2
    assert np.array_equal(buf.states[1], s1)
    assert np.array_equal(buf.next_states[1], s2)
    assert buf.terminations[1][0] == 1.0


def test_override_action_replaces_stored_action_reward_passed_through_unchanged():
    """The policy no longer applies any intervention penalty itself -- reward
    is stored exactly as given, since that's now the caller's responsibility."""
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM, min_buffer_size=10_000_000, batch_size=4)

    s0 = np.zeros(STATE_DIM, dtype=np.float32)
    s1 = np.ones(STATE_DIM, dtype=np.float32)
    proposed_action = np.array([9.0, 9.0], dtype=np.float32)
    human_action = np.array([-9.0, -9.0], dtype=np.float32)

    policy.store_transition(s0, proposed_action, reward=0.5, post_state=s1, done=False, override_action=human_action)

    buf = policy._online_buffer
    assert np.array_equal(buf.actions[0], human_action)
    assert buf.rewards[0][0] == 0.5


def test_update_reduces_q_loss_without_nan():
    torch.manual_seed(0)
    np.random.seed(0)
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM, min_buffer_size=32, batch_size=32, utd_ratio=1)

    n = 200
    policy._online_buffer.add(
        np.random.randn(n, STATE_DIM).astype(np.float32),
        np.random.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32),
        np.random.randn(n, 1).astype(np.float32),
        np.random.randn(n, STATE_DIM).astype(np.float32),
        np.zeros((n, 1), dtype=np.float32),
    )

    losses = []
    for _ in range(200):
        batch = policy._sample_mixed_batch(32)
        states_t = torch.from_numpy(batch.states)
        actions_t = torch.from_numpy(batch.actions)
        rewards_t = torch.from_numpy(batch.rewards)
        next_states_t = torch.from_numpy(batch.next_states)
        terminations_t = torch.from_numpy(batch.terminations)
        with torch.no_grad():
            next_actions, next_log_probs = policy._policy.sample(next_states_t)
            alpha = policy._log_alpha.exp().detach()
            target_q = torch.min(
                policy._q1_target(next_states_t, next_actions), policy._q2_target(next_states_t, next_actions)
            )
            target = rewards_t + policy._gamma * (1.0 - terminations_t) * (target_q - alpha * next_log_probs)
        loss = nn.functional.mse_loss(policy._q1(states_t, actions_t), target).item()
        losses.append(loss)
        policy._update(batch)

    assert not any(np.isnan(loss) for loss in losses)
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_mixed_batch_falls_back_to_pure_online_when_demo_buffer_empty():
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM, min_buffer_size=10_000_000, batch_size=8)
    policy._online_buffer.add(
        np.random.randn(50, STATE_DIM).astype(np.float32),
        np.random.uniform(-1, 1, (50, ACTION_DIM)).astype(np.float32),
        np.zeros((50, 1), dtype=np.float32),
        np.random.randn(50, STATE_DIM).astype(np.float32),
        np.zeros((50, 1), dtype=np.float32),
    )
    batch = policy._sample_mixed_batch(8)
    assert batch.states.shape == (8, STATE_DIM)


def test_mixed_batch_splits_50_50_between_online_and_demo():
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM, min_buffer_size=10_000_000, batch_size=8)
    actions = np.random.uniform(-1, 1, (50, ACTION_DIM)).astype(np.float32)
    rewards = np.zeros((50, 1), dtype=np.float32)
    next_states = np.random.randn(50, STATE_DIM).astype(np.float32)
    terminations = np.zeros((50, 1), dtype=np.float32)

    policy._online_buffer.add(np.random.randn(50, STATE_DIM).astype(np.float32), actions, rewards, next_states, terminations)
    demo_states = np.full((50, STATE_DIM), 999.0, dtype=np.float32)  # marker value
    policy.add_demo_transitions(demo_states, actions, rewards, next_states, terminations)

    batch = policy._sample_mixed_batch(8)
    num_from_demo = np.sum(np.all(batch.states == 999.0, axis=1))
    assert num_from_demo == 4


def test_checkpoint_round_trip_produces_identical_actions(tmp_path: Path):
    torch.manual_seed(0)
    policy = RLPDPolicy(STATE_DIM, ACTION_DIM)
    checkpoint_path = tmp_path / "rlpd.pt"
    policy.save_checkpoint(checkpoint_path)
    loaded = RLPDPolicy.load_from_checkpoint(data_location=tmp_path, checkpoint_location=checkpoint_path)

    test_state = np.random.randn(STATE_DIM).astype(np.float32)
    torch.manual_seed(42)
    action_before = policy.act(test_state)
    torch.manual_seed(42)
    action_after = loaded.act(test_state)
    assert np.allclose(action_before, action_after, atol=1e-5)
