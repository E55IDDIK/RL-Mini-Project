"""Double DQN implementation for Dynamic CVRP."""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_KEYS = ("node_feats", "dist", "traffic_obs", "traffic_mask", "vehicle", "action_mask")


def make_target_network(online: nn.Module) -> nn.Module:
    """Create a frozen copy of the online network for Double DQN."""
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


def sync_target_network(online: nn.Module, target: nn.Module) -> None:
    """Copy online network weights to target network."""
    target.load_state_dict(online.state_dict())


def select_action(
    online: nn.Module,
    obs: dict,
    epsilon: float,
    rng: np.random.Generator,
    n_nodes: int,
) -> int:
    """Epsilon-greedy action selection among valid actions."""
    mask = obs["action_mask"]
    valid_actions = np.flatnonzero(mask > 0)

    if valid_actions.size == 0:
        raise RuntimeError("select_action called with no valid actions.")

    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))

    device = next(online.parameters()).device
    obs_t = {k: torch.as_tensor(obs[k], device=device).unsqueeze(0) for k in OBS_KEYS}

    online.eval()
    with torch.no_grad():
        q = online(obs_t)
        masked_q = q.masked_fill(obs_t["action_mask"] == 0, float("-inf"))
        action = int(masked_q.argmax(dim=1).item())
    online.train()

    return action


def double_dqn_loss(
    online: nn.Module,
    target: nn.Module,
    batch: dict,
    gamma: float,
) -> torch.Tensor:
    """Compute Double DQN Huber loss on a sampled batch."""
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    action = batch["action"]
    reward = batch["reward"]
    terminated = batch["terminated"]

    # Q-values for taken actions
    q_values = online(obs)
    q_taken = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Action selection via online net (masked to valid actions)
        next_q_online = online(next_obs)
        next_mask = next_obs["action_mask"]
        next_q_online_masked = next_q_online.masked_fill(next_mask == 0, float("-inf"))
        best_actions = next_q_online_masked.argmax(dim=1)

        # Action evaluation via target net
        next_q_target = target(next_obs)
        next_q_target_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        # Bellman target (only terminated zeroes the bootstrap, not truncated)
        not_done = 1.0 - terminated
        y = reward + gamma * not_done * next_q_target_selected

    return F.smooth_l1_loss(q_taken, y)
