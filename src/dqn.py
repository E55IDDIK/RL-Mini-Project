"""Double DQN core: stub network, masking helpers, loss, target sync.

Matches the interface contract in the briefing exactly:
- network(obs) -> Q-values, shape (N,) for a single obs, (B, N) for a batch
- the network does NOT mask illegal actions -- every argmax/max here does
- no PyG Batch/DataLoader machinery -- batching is a plain leading dim
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class StubQNetwork(nn.Module):
    """Placeholder with the exact real network's I/O contract. Ignores the
    observation content entirely -- only shape-compatible, not meaningful.
    Delete this once the real GNN encoder + GRU + DQN head is ready."""

    def __init__(self, n_nodes: int):
        super().__init__()
        self.n_nodes = n_nodes
        self._dummy = nn.Linear(1, n_nodes)  # gives it real, trainable params

    def forward(self, obs_batch: dict) -> torch.Tensor:
        vehicle = obs_batch["vehicle"]
        single = vehicle.dim() == 1
        if single:
            vehicle = vehicle.unsqueeze(0)
        b = vehicle.shape[0]
        out = self._dummy(torch.ones(b, 1, device=vehicle.device))
        return out.squeeze(0) if single else out


def masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """argmax over q_values with illegal actions (mask == 0) forced to -inf.

    q_values, action_mask: (B, N) -> returns (B,) int64 action indices.
    """
    masked_q = q_values.masked_fill(action_mask == 0, float("-inf"))
    return masked_q.argmax(dim=1)


def masked_max(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """max over q_values with illegal actions forced to -inf. Returns (B,)."""
    masked_q = q_values.masked_fill(action_mask == 0, float("-inf"))
    return masked_q.max(dim=1).values


def select_action(network: nn.Module, obs: dict, epsilon: float, rng, n_actions: int) -> int:
    """Epsilon-greedy action selection for a single (unbatched) observation.

    With probability epsilon, sample uniformly among *legal* actions (never a
    fully random action over all N -- exploring into illegal moves would just
    raise in env.step). Otherwise act greedily w.r.t. the online network,
    masked to legal actions.
    """
    action_mask = obs["action_mask"]
    if rng.random() < epsilon:
        legal_idx = [i for i, m in enumerate(action_mask) if m > 0]
        return int(rng.choice(legal_idx))

    with torch.no_grad():
        device = next(network.parameters()).device
        obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
        q = network(obs_t)  # (1, N)
        mask_t = obs_t["action_mask"]
        action = masked_argmax(q, mask_t)
    return int(action.item())


def sync_target_network(online: nn.Module, target: nn.Module) -> None:
    """Hard copy: target <- online. Call every `target_update_freq` steps."""
    target.load_state_dict(online.state_dict())


def make_target_network(online: nn.Module) -> nn.Module:
    """Build a frozen copy of `online` to serve as the initial target network."""
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


def double_dqn_loss(online: nn.Module, target: nn.Module, batch: dict, gamma: float) -> torch.Tensor:
    """Double DQN loss (Huber), per Section 6 of the briefing:

        a*     = argmax_a Q_online(next_obs, a)   [masked to next_obs's legal actions]
        target = reward + gamma * Q_target(next_obs, a*) * (1 - terminated)
        loss   = huber_loss(Q_online(obs, action), target)

    Critical details this deliberately gets right:
    - action selection (a*) uses the ONLINE network; its value is looked up
      in the TARGET network -- never Q_target for both (that's vanilla DQN).
    - masking for both the argmax and the value lookup uses
      next_obs["action_mask"], not obs["action_mask"].
    - the episode-end flag is `terminated` ONLY. `truncated` is an artificial
      cutoff, not a real end of the world, so the agent must still bootstrap
      through it. `batch["truncated"]` is intentionally unused here.
    """
    obs, next_obs = batch["obs"], batch["next_obs"]
    action = batch["action"]
    reward = batch["reward"]
    terminated = batch["terminated"]

    # Q(obs, action) from the online network.
    q_online_obs = online(obs)  # (B, N)
    q_taken = q_online_obs.gather(1, action.unsqueeze(1)).squeeze(1)  # (B,)

    with torch.no_grad():
        next_mask = next_obs["action_mask"]

        # a* selected by the ONLINE network, masked to next_obs's legal actions.
        q_online_next = online(next_obs)  # (B, N)
        a_star = masked_argmax(q_online_next, next_mask)  # (B,)

        # a*'s value looked up in the TARGET network (also masked, for
        # consistency/robustness -- a* is already legal so this is a no-op
        # on the selected index, but keeps -inf out of the gather if a*
        # somehow ties into an illegal slot due to numerical edge cases).
        q_target_next = target(next_obs)  # (B, N)
        q_target_next = q_target_next.masked_fill(next_mask == 0, float("-inf"))
        next_value = q_target_next.gather(1, a_star.unsqueeze(1)).squeeze(1)  # (B,)

        # At a genuine terminal state, next_obs's action_mask is legitimately
        # all-zero (nowhere left to go -- see env.py's _valid_mask), so
        # next_value can be -inf there. That's fine numerically as long as we
        # never multiply -inf by the 0 bootstrap factor (inf * 0 = nan), so
        # sanitize non-finite values to 0 before the bootstrap multiply --
        # `bootstrap` already guarantees terminated transitions contribute 0.
        next_value = torch.nan_to_num(next_value, neginf=0.0, posinf=0.0, nan=0.0)

        # terminated ONLY -- truncation still bootstraps. See docstring.
        bootstrap = 1.0 - terminated
        td_target = reward + gamma * next_value * bootstrap

    return F.huber_loss(q_taken, td_target)