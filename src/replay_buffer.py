"""Replay buffer for GNN-DQN.

Stores full observation dicts (Section 3 of the briefing) as plain numpy
arrays -- no PyTorch Geometric Data/Batch objects, no flattening. The graph
is a fixed-size complete graph (N nodes) every episode, so a leading batch
dimension on ordinary arrays is all batching ever needs.

Per transition: (obs, action, reward, next_obs, terminated, truncated).
`terminated` and `truncated` are stored separately and never combined here --
only the loss function decides how to use them (see dqn.py / Section 3's
critical correctness detail: only `terminated` should zero the bootstrap).
"""

from __future__ import annotations

import numpy as np
import torch

# Keys every observation dict must have, per the interface contract (Section 3).
OBS_KEYS = ("node_feats", "dist", "traffic_obs", "traffic_mask", "vehicle", "action_mask")


class ReplayBuffer:
    """Fixed-capacity circular buffer of (obs, action, reward, next_obs, terminated, truncated)."""

    def __init__(self, capacity: int, n_nodes: int, node_feat_dim: int = 8, seed: int | None = None):
        self.capacity = int(capacity)
        self.n_nodes = int(n_nodes)
        self.node_feat_dim = int(node_feat_dim)
        self.rng = np.random.default_rng(seed)

        N = self.n_nodes
        # Pre-allocated arrays, one slot per stored transition -- avoids
        # per-step Python object churn and makes sampling a cheap fancy-index.
        self._obs = {
            "node_feats": np.zeros((self.capacity, N, self.node_feat_dim), dtype=np.float32),
            "dist": np.zeros((self.capacity, N, N), dtype=np.float32),
            "traffic_obs": np.zeros((self.capacity, N, N), dtype=np.float32),
            "traffic_mask": np.zeros((self.capacity, N, N), dtype=np.float32),
            "vehicle": np.zeros((self.capacity, 3), dtype=np.float32),
            "action_mask": np.zeros((self.capacity, N), dtype=np.float32),
        }
        self._next_obs = {k: np.zeros_like(v) for k, v in self._obs.items()}
        self._action = np.zeros((self.capacity,), dtype=np.int64)
        self._reward = np.zeros((self.capacity,), dtype=np.float32)
        self._terminated = np.zeros((self.capacity,), dtype=np.float32)
        self._truncated = np.zeros((self.capacity,), dtype=np.float32)

        self._write_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: dict, action: int, reward: float, next_obs: dict,
            terminated: bool, truncated: bool) -> None:
        i = self._write_idx
        for k in OBS_KEYS:
            self._obs[k][i] = obs[k]
            self._next_obs[k][i] = next_obs[k]
        self._action[i] = action
        self._reward[i] = reward
        self._terminated[i] = float(terminated)
        self._truncated[i] = float(truncated)

        self._write_idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> dict:
        """Return a batch as a dict of torch tensors, each with a leading batch dim.

        Shape: {"obs": {...}, "action": (B,), "reward": (B,),
                "next_obs": {...}, "terminated": (B,), "truncated": (B,)}
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)

        obs_batch = {k: torch.as_tensor(self._obs[k][idx], device=device) for k in OBS_KEYS}
        next_obs_batch = {k: torch.as_tensor(self._next_obs[k][idx], device=device) for k in OBS_KEYS}

        return {
            "obs": obs_batch,
            "action": torch.as_tensor(self._action[idx], device=device),
            "reward": torch.as_tensor(self._reward[idx], device=device),
            "next_obs": next_obs_batch,
            "terminated": torch.as_tensor(self._terminated[idx], device=device),
            "truncated": torch.as_tensor(self._truncated[idx], device=device),
        }