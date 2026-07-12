"""RLPDPolicy: a HILPolicy (see policy.py) implementing RLPD (Reinforcement
Learning with Prior Data, Ball et al. 2023) -- a high-update-to-data-ratio
off-policy actor-critic built on standard SAC:

  - twin Q-networks (Q1, Q2), each with a Polyak-averaged target copy, using
    the *minimum* of the two targets to form the TD target (clipped double-Q,
    the standard SAC/TD3 fix for Q overestimation bias -- important here
    since overestimation compounds faster at a high UTD ratio)
  - a single tanh-squashed Gaussian policy over the full (bounded continuous)
    action space -- Unity only ever accepts bounded-continuous commands (see
    RemoteJointController's Lerp-based joint targets), so there's no
    structural need for a separate discrete gripper head
  - automatic entropy-coefficient (alpha) tuning, the modern standard over a
    fixed entropy temperature

RLPD's own contribution over vanilla SAC: every training batch is drawn half
from a static offline buffer (pre-loaded once from recorded demonstrations)
and half from a growing online buffer (see replay_buffer.ReplayBuffer for
both), which is what lets it train stably at a much higher UTD ratio than
plain SAC.

Human-in-the-loop specifics: store_transition() takes one complete (pre_state,
action, reward, post_state, done) transition per call. If a human intervened
(override_action given), the transition's stored action is what was actually
executed instead of what the policy proposed -- reward is expected to already
reflect that (e.g. an intervention penalty) by the time it's passed in; this
class just records it, it doesn't compute or adjust it.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from replay_buffer import Batch, ReplayBuffer

HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005  # Polyak averaging coefficient for target network soft updates
UTD_RATIO = 1  # gradient steps per store_transition call, once there's enough data
MIN_BUFFER_SIZE = 256  # don't start training until the online buffer has at least this many transitions
BATCH_SIZE = 256
ONLINE_BATCH_FRACTION = 0.5  # RLPD's 50/50 online/offline split
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class QNetwork(nn.Module):
    """LayerNorm after each hidden layer isn't just a nice-to-have here -- RLPD
    (Ball et al. 2023) identifies it as specifically what prevents catastrophic
    Q-value overestimation from bootstrapped TD targets compounding over many
    gradient steps, particularly when terminal (zero-bootstrap) transitions are
    rare relative to the total number of updates -- exactly the sparse-success
    pick-and-place setting this critic trains on."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy (the standard SAC actor): samples a raw
    action from N(mean, std), squashes it through tanh to bound it to
    [-1, 1], and returns the squashed action's log-probability with the
    standard tanh Jacobian correction (SAC paper, appendix C, eq 21)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(state)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()  # reparameterized: differentiable through the sample
        action = torch.tanh(raw_action)

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class RLPDPolicy:
    """See module docstring. Satisfies HILPolicy (policy.py) structurally.

    The demo (offline) buffer starts empty -- use add_demo_transitions() to
    populate it (e.g. derived from recorded teleop sessions) once, separately
    from load_from_checkpoint, which only restores trained network weights.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        online_capacity: int = 100_000,
        demo_capacity: int = 100_000,
        hidden_size: int = HIDDEN_SIZE,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        tau: float = TAU,
        utd_ratio: int = UTD_RATIO,
        batch_size: int = BATCH_SIZE,
        min_buffer_size: int = MIN_BUFFER_SIZE,
        online_batch_fraction: float = ONLINE_BATCH_FRACTION,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self._gamma = gamma
        self._tau = tau
        self._utd_ratio = utd_ratio
        self._batch_size = batch_size
        self._min_buffer_size = min_buffer_size
        self._online_batch_fraction = online_batch_fraction

        self._policy = GaussianPolicy(state_dim, action_dim, hidden_size)
        self._q1 = QNetwork(state_dim, action_dim, hidden_size)
        self._q2 = QNetwork(state_dim, action_dim, hidden_size)
        self._q1_target = QNetwork(state_dim, action_dim, hidden_size)
        self._q2_target = QNetwork(state_dim, action_dim, hidden_size)
        self._q1_target.load_state_dict(self._q1.state_dict())
        self._q2_target.load_state_dict(self._q2.state_dict())

        self._log_alpha = torch.zeros(1, requires_grad=True)
        self._target_entropy = -float(action_dim)  # standard heuristic (SAC follow-up paper)

        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=learning_rate)
        self._q_optimizer = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()), lr=learning_rate
        )
        self._alpha_optimizer = torch.optim.Adam([self._log_alpha], lr=learning_rate)

        self._online_buffer = ReplayBuffer(online_capacity, state_dim, action_dim)
        self._demo_buffer = ReplayBuffer(demo_capacity, state_dim, action_dim)

    def add_demo_transitions(
        self,
        states: npt.NDArray[np.float32],
        actions: npt.NDArray[np.float32],
        rewards: npt.NDArray[np.float32],
        next_states: npt.NDArray[np.float32],
        terminations: npt.NDArray[np.float32],
    ) -> None:
        """Populate the static offline/demo buffer -- see replay_buffer.ReplayBuffer.add
        for the expected (num_samples, d) shapes."""
        self._demo_buffer.add(states, actions, rewards, next_states, terminations)

    def act(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _ = self._policy.sample(state_tensor)
        return action.squeeze(0).numpy()

    def estimate_q(self, state: npt.NDArray[np.float64], action: npt.NDArray[np.float64]) -> float:
        """Q-value estimate for one (state, action) pair -- the same clipped
        double-Q minimum used to form the SAC/RLPD target (see _update), exposed
        for logging/monitoring rather than anything training-critical (e.g.
        watching whether actions that trigger human intervention already look
        low-value to the critic)."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            q = torch.min(self._q1(state_tensor, action_tensor), self._q2(state_tensor, action_tensor))
        return float(q.item())

    def store_transition(
        self,
        pre_state: npt.NDArray[np.float64],
        action: npt.NDArray[np.float64],
        reward: float,
        post_state: npt.NDArray[np.float64],
        done: bool,
        override_action: npt.NDArray[np.float64] | None = None,
    ) -> None:
        stored_action = action if override_action is None else override_action
        self._push_to_online_buffer(pre_state, stored_action, reward, next_state=post_state, done=done)
        self._train_if_ready()

    def _push_to_online_buffer(
        self,
        state: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32],
        reward: float,
        next_state: npt.NDArray[np.float32],
        done: bool,
    ) -> None:
        self._online_buffer.add(
            states=np.asarray(state, dtype=np.float32).reshape(1, -1),
            actions=np.asarray(action, dtype=np.float32).reshape(1, -1),
            rewards=np.array([[reward]], dtype=np.float32),
            next_states=np.asarray(next_state, dtype=np.float32).reshape(1, -1),
            terminations=np.array([[float(done)]], dtype=np.float32),
        )

    def _sample_mixed_batch(self, batch_size: int) -> Batch:
        """RLPD's mixing scheme: online_batch_fraction from the online buffer,
        the rest from the demo buffer -- falling back to 100% online if the
        demo buffer hasn't been populated (add_demo_transitions never called)."""
        if len(self._demo_buffer) == 0:
            return self._online_buffer.sample(batch_size)

        online_n = min(max(int(round(batch_size * self._online_batch_fraction)), 0), batch_size)
        demo_n = batch_size - online_n
        online_batch = self._online_buffer.sample(online_n) if online_n > 0 else None
        demo_batch = self._demo_buffer.sample(demo_n) if demo_n > 0 else None

        if online_batch is None:
            return demo_batch
        if demo_batch is None:
            return online_batch
        return Batch(
            states=np.concatenate([online_batch.states, demo_batch.states]),
            actions=np.concatenate([online_batch.actions, demo_batch.actions]),
            rewards=np.concatenate([online_batch.rewards, demo_batch.rewards]),
            next_states=np.concatenate([online_batch.next_states, demo_batch.next_states]),
            terminations=np.concatenate([online_batch.terminations, demo_batch.terminations]),
        )

    def _train_if_ready(self) -> None:
        if len(self._online_buffer) < self._min_buffer_size:
            return
        for _ in range(self._utd_ratio):
            self._update(self._sample_mixed_batch(self._batch_size))

    def _update(self, batch: Batch) -> None:
        states = torch.from_numpy(batch.states)
        actions = torch.from_numpy(batch.actions)
        rewards = torch.from_numpy(batch.rewards)
        next_states = torch.from_numpy(batch.next_states)
        terminations = torch.from_numpy(batch.terminations)

        alpha = self._log_alpha.exp().detach()

        # --- Q loss (clipped double-Q target) ---
        with torch.no_grad():
            next_actions, next_log_probs = self._policy.sample(next_states)
            target_q = torch.min(self._q1_target(next_states, next_actions), self._q2_target(next_states, next_actions))
            target = rewards + self._gamma * (1.0 - terminations) * (target_q - alpha * next_log_probs)

        q1_loss = nn.functional.mse_loss(self._q1(states, actions), target)
        q2_loss = nn.functional.mse_loss(self._q2(states, actions), target)
        q_loss = q1_loss + q2_loss

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        # --- policy loss ---
        new_actions, log_probs = self._policy.sample(states)
        q_new = torch.min(self._q1(states, new_actions), self._q2(states, new_actions))
        policy_loss = (alpha * log_probs - q_new).mean()

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # --- alpha loss (automatic entropy tuning) ---
        alpha_loss = -(self._log_alpha.exp() * (log_probs.detach() + self._target_entropy)).mean()

        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()

        # --- soft-update targets ---
        self._soft_update(self._q1, self._q1_target)
        self._soft_update(self._q2, self._q2_target)

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(target.parameters(), online.parameters()):
                target_param.mul_(1.0 - self._tau).add_(online_param, alpha=self._tau)

    def save_checkpoint(self, checkpoint_location: Path) -> None:
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_size": self.hidden_size,
                "policy_state_dict": self._policy.state_dict(),
                "q1_state_dict": self._q1.state_dict(),
                "q2_state_dict": self._q2.state_dict(),
                "q1_target_state_dict": self._q1_target.state_dict(),
                "q2_target_state_dict": self._q2_target.state_dict(),
                "log_alpha": self._log_alpha.detach(),
            },
            checkpoint_location,
        )

    @classmethod
    def load_from_checkpoint(cls, data_location: Path, checkpoint_location: Path) -> "RLPDPolicy":
        """Restores trained network weights from checkpoint_location. Does NOT
        touch the demo buffer -- data_location is unused here; call
        add_demo_transitions() separately to populate it (e.g. from recorded
        teleop sessions), since that's a data-ingestion concern distinct from
        restoring a trained checkpoint."""
        del data_location
        checkpoint = torch.load(checkpoint_location, weights_only=False)
        policy = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_size=checkpoint["hidden_size"],
        )
        policy._policy.load_state_dict(checkpoint["policy_state_dict"])
        policy._q1.load_state_dict(checkpoint["q1_state_dict"])
        policy._q2.load_state_dict(checkpoint["q2_state_dict"])
        policy._q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        policy._q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        with torch.no_grad():
            policy._log_alpha.copy_(checkpoint["log_alpha"])
        return policy
