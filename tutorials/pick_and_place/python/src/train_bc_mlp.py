"""Experiment 7: train a small MLP via behavior cloning on recorded teleop
(state, action) pairs.

Loads every states_*.npy / actions_*.npy pair in the teleop directory,
concatenates them into one training set, and trains a plain MLP to regress
action from state. This dataset is small enough that a torch Dataset/
DataLoader would just be ceremony -- the arrays are held in memory whole and
indexed directly to build minibatches.

Self-contained for now (per-experiment file); shared training tooling (across
future experiments/models) can get factored out once there's a second
training script that actually needs it.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

TELEOP_DIRECTORY = Path("/Users/ztcecere/CodeRepository/Unity-Robotics-Hub/teleop")
MODEL_SAVE_PATH = TELEOP_DIRECTORY / "bc_mlp.pt"

HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 200
VALIDATION_FRACTION = 0.1


def load_dataset(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate every states_*.npy / actions_*.npy pair in directory."""
    states_list = []
    actions_list = []
    for states_path in sorted(directory.glob("states_*.npy")):
        timestamp = states_path.stem.removeprefix("states_")
        actions_path = directory / f"actions_{timestamp}.npy"
        if not actions_path.exists():
            print(f"Skipping {states_path.name}: no matching {actions_path.name}")
            continue
        states_list.append(np.load(states_path))
        actions_list.append(np.load(actions_path))

    if not states_list:
        raise FileNotFoundError(f"No states_*.npy / actions_*.npy pairs found in {directory}")

    states = np.concatenate(states_list, axis=0).astype(np.float32)
    actions = np.concatenate(actions_list, axis=0).astype(np.float32)
    return states, actions


class Normalizer(nn.Module):
    """Standardizes inputs with a fixed mean/std computed once, up front, from the
    training set. mean/std are registered as buffers rather than parameters, so
    they're saved/loaded with the rest of the model's state_dict() (and moved
    along with .to(device)) but never touched by the optimizer."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class BehaviorCloningMLP(nn.Module):
    """state_mean/state_std default to zeros/ones (i.e. a no-op normalizer) --
    that's for reconstructing an empty model of the right shape before calling
    load_state_dict(), which then overwrites them with the real saved buffers.
    Pass the real, computed values when actually training a new model.

    pause_timesteps mirrors Replay.pause_timesteps (see policy.py): act() only
    runs a fresh forward pass every (pause_timesteps + 1) calls, repeating the
    previous prediction in between, rather than a new prediction every single
    call. This matches inference-time query cadence to the sparsity of the
    recorded training data (see teleop_controller.RECORD_ONLY_NONZERO_ACTIONS),
    rather than affecting training/forward() at all -- only act() paces itself."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = HIDDEN_SIZE,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        pause_timesteps: int = 2,
    ):
        super().__init__()
        if state_mean is None:
            state_mean = np.zeros(state_dim, dtype=np.float32)
        if state_std is None:
            state_std = np.ones(state_dim, dtype=np.float32)
        self.normalizer = Normalizer(state_mean, state_std)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),  # every action component is in [-1, 1]
        )
        self._pause_timesteps = pause_timesteps
        self._repeats_remaining = 0
        self._last_action: npt.NDArray[np.float64] | None = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(state))

    @classmethod
    def load_from_checkpoint(cls, data_location: Path, checkpoint_location: Path) -> "BehaviorCloningMLP":
        """Satisfies the Policy interface (see policy.py): data_location is unused
        here since checkpoint_location's file already self-contains architecture
        dims and normalizer stats -- no separate data needed to reconstruct it."""
        del data_location
        checkpoint = torch.load(checkpoint_location, weights_only=False)
        model = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_size=checkpoint["hidden_size"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def act(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Satisfies the Policy interface (see policy.py): numpy state in, numpy
        action out, wrapping forward()'s torch tensors. See pause_timesteps in
        the class docstring for why this doesn't predict fresh every call."""
        if self._repeats_remaining > 0:
            self._repeats_remaining -= 1
            return self._last_action

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = self(state_tensor).squeeze(0)
        self._last_action = action_tensor.numpy()
        self._repeats_remaining = self._pause_timesteps
        return self._last_action


def iterate_batches(
    states: np.ndarray, actions: np.ndarray, batch_size: int, shuffle: bool = True
):
    """Yield one epoch's worth of (state_batch, action_batch) tensors."""
    num_examples = len(states)
    indices = np.random.permutation(num_examples) if shuffle else np.arange(num_examples)
    for start in range(0, num_examples, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield torch.from_numpy(states[batch_indices]), torch.from_numpy(actions[batch_indices])


def main() -> None:
    states, actions = load_dataset(TELEOP_DIRECTORY)
    print(f"Loaded {len(states)} (state, action) pairs from {TELEOP_DIRECTORY}")

    # Positions (meters), quaternions (unit-norm), gripper width (meters), and
    # joint angles (radians) are on very different scales, which matters a lot
    # for training -- computed once here and baked into the model's own
    # Normalizer layer, rather than tracked/applied separately at inference
    # time. Actions are already all in [-1, 1], so they're left as-is.
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-6

    num_validation = max(1, int(len(states) * VALIDATION_FRACTION))
    permutation = np.random.permutation(len(states))
    validation_indices, train_indices = permutation[:num_validation], permutation[num_validation:]

    train_states, train_actions = states[train_indices], actions[train_indices]
    val_states = torch.from_numpy(states[validation_indices])
    val_actions = torch.from_numpy(actions[validation_indices])

    model = BehaviorCloningMLP(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        state_mean=state_mean,
        state_std=state_std,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for state_batch, action_batch in iterate_batches(train_states, train_actions, BATCH_SIZE):
            optimizer.zero_grad()
            predicted = model(state_batch)
            loss = loss_fn(predicted, action_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(val_states), val_actions).item()
            print(f"epoch {epoch:4d}  train_loss={epoch_loss / num_batches:.5f}  val_loss={val_loss:.5f}")

    torch.save(
        {
            # state_mean/state_std are not saved separately -- they're already
            # in here as model_state_dict["normalizer.mean"/"normalizer.std"].
            "model_state_dict": model.state_dict(),
            "hidden_size": HIDDEN_SIZE,
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
        },
        MODEL_SAVE_PATH,
    )
    print(f"Saved trained model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
