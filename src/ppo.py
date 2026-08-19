"""GNN-PPO (Actor-Critic) for the Dynamic CVRP environment.

Design notes (read this before touching the architecture):

1. Shared GNN encoder with DQN.
   The GNNEncoder from gnn_encoder.py is reused as-is. It produces per-node
   embeddings (N, hidden_dim) and a graph_summary vector. The Actor head
   scores each node individually (same technique as GNNQNetwork), while the
   Critic head maps the graph_summary to a scalar V(s).

2. No cross-step recurrent hidden state.
   Same reasoning as the DQN (see gnn_encoder.py docstring): the env's
   cumulative observation mask already externalizes the belief state, so a
   stateless forward pass is correct.

3. terminated vs truncated -- critical for GAE.
   The rollout buffer stores terminated and truncated separately.  GAE uses
   ONLY terminated to zero out the value bootstrap:
       delta = reward + gamma * V(s') * (1 - terminated) - V(s)
   Truncated episodes still bootstrap from V(s') because the episode was
   cut short artificially (max_steps), not because a true terminal state
   was reached.  This matches the DQN's treatment in dqn.py / replay_buffer.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.gnn_encoder import (
    GNNEncoder,
    build_complete_edge_index,
    build_edge_attr,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)

OBS_KEYS = ("node_feats", "dist", "traffic_obs", "traffic_mask", "vehicle", "action_mask")


# ============================================================================
# Actor-Critic network
# ============================================================================

class GNNActorCritic(nn.Module):
    """GNN-based Actor-Critic for PPO.

    Forward contract (matches how train_ppo.py calls it):
        action, log_prob, entropy, value = ac.get_action_and_value(obs_batch)
        value                            = ac.get_value(obs_batch)

    obs_batch values have a leading batch dimension (B, ...).
    """

    def __init__(
        self,
        n_nodes: int,
        node_feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 32,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        head_hidden: int = 32,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder = GNNEncoder(node_feat_dim, hidden_dim, edge_feat_dim)
        self.register_buffer("edge_index", build_complete_edge_index(n_nodes))

        # Actor head: scores each node using [node_emb | graph_summary]
        actor_input_dim = hidden_dim + self.encoder.summary_dim
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Critic head: maps graph_summary -> scalar V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(self.encoder.summary_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def _encode_batch(self, obs_batch: dict):
        """Run the GNN encoder on a batch and return per-node embeddings,
        graph summaries, and action masks.

        Returns:
            all_x:       list[Tensor] of length B, each (N, hidden_dim)
            all_summary: list[Tensor] of length B, each (summary_dim,)
            mask:        Tensor (B, N)
        """
        node_feats = obs_batch["node_feats"]      # (B, N, F)
        dist = obs_batch["dist"]                  # (B, N, N)
        traffic_obs = obs_batch["traffic_obs"]    # (B, N, N)
        traffic_mask = obs_batch["traffic_mask"]  # (B, N, N)
        vehicle = obs_batch["vehicle"]            # (B, 3)
        mask = obs_batch["action_mask"]           # (B, N)

        B = node_feats.shape[0]
        edge_index = self.edge_index

        all_x = []
        all_summary = []
        for b in range(B):
            edge_attr = build_edge_attr(
                dist[b], traffic_obs[b], traffic_mask[b], edge_index,
            )
            summary, x = self.encoder(node_feats[b], edge_index, edge_attr, vehicle[b])
            all_x.append(x)
            all_summary.append(summary)

        return all_x, all_summary, mask

    def _logits_and_value(self, obs_batch: dict):
        """Compute raw actor logits (B, N) and critic value (B, 1)."""
        all_x, all_summary, mask = self._encode_batch(obs_batch)
        B = len(all_x)

        logits_list = []
        values_list = []
        for b in range(B):
            context = all_summary[b].unsqueeze(0).expand(self.n_nodes, -1)
            x_aug = torch.cat([all_x[b], context], dim=-1)
            logits_list.append(self.actor_head(x_aug).squeeze(-1))  # (N,)
            values_list.append(self.critic_head(all_summary[b]))     # (1,)

        logits = torch.stack(logits_list, dim=0)  # (B, N)
        values = torch.stack(values_list, dim=0)  # (B, 1)

        # action masking: set invalid actions to -inf so they get zero
        # probability from the Categorical
        logits = logits.masked_fill(mask == 0, float("-inf"))

        return logits, values, mask

    def get_action_and_value(
        self, obs_batch: dict, action: torch.Tensor | None = None,
    ):
        """Sample an action (or evaluate a given action) and return
        (action, log_prob, entropy, value).

        During rollout collection: action=None -> sample from the policy.
        During PPO update: action=given -> evaluate that action's log_prob.
        """
        logits, values, mask = self._logits_and_value(obs_batch)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values.squeeze(-1)

    def get_value(self, obs_batch: dict) -> torch.Tensor:
        """Return V(s) only, shape (B,).  Used for bootstrapping at the end
        of a rollout."""
        _, values, _ = self._logits_and_value(obs_batch)
        return values.squeeze(-1)


# ============================================================================
# On-policy rollout buffer with GAE
# ============================================================================

class PPORolloutBuffer:
    """Fixed-length buffer that stores one full rollout (spanning potentially
    multiple episodes), then computes GAE advantages and returns.

    Storage layout: flat arrays of length `rollout_length`.  Each index t
    holds the transition (obs_t, action_t, reward_t, terminated_t,
    truncated_t, log_prob_t, value_t).

    CRITICAL: terminated vs truncated
    ---------------------------------
    GAE uses ONLY terminated to decide whether to zero the bootstrap:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
    Truncated episodes still bootstrap from V(s_{t+1}) because the episode
    was artificially cut (max_steps hit), not because a true absorbing state
    was reached.  This matches the DQN code's convention exactly.
    """

    def __init__(self, rollout_length: int, n_nodes: int, node_feat_dim: int = 8):
        self.rollout_length = rollout_length
        self.n_nodes = n_nodes
        self.node_feat_dim = node_feat_dim
        self.pos = 0  # write cursor

        N = n_nodes
        L = rollout_length

        # observation arrays
        self.obs = {
            "node_feats":   np.zeros((L, N, node_feat_dim), dtype=np.float32),
            "dist":         np.zeros((L, N, N), dtype=np.float32),
            "traffic_obs":  np.zeros((L, N, N), dtype=np.float32),
            "traffic_mask": np.zeros((L, N, N), dtype=np.float32),
            "vehicle":      np.zeros((L, 3), dtype=np.float32),
            "action_mask":  np.zeros((L, N), dtype=np.float32),
        }

        self.actions     = np.zeros(L, dtype=np.int64)
        self.rewards     = np.zeros(L, dtype=np.float32)
        self.terminated  = np.zeros(L, dtype=np.float32)
        self.truncated   = np.zeros(L, dtype=np.float32)
        self.log_probs   = np.zeros(L, dtype=np.float32)
        self.values      = np.zeros(L, dtype=np.float32)

        # computed after the rollout is complete
        self.advantages  = np.zeros(L, dtype=np.float32)
        self.returns     = np.zeros(L, dtype=np.float32)

    def add(self, obs: dict, action: int, reward: float, terminated: bool,
            truncated: bool, log_prob: float, value: float) -> None:
        i = self.pos
        for k in OBS_KEYS:
            self.obs[k][i] = obs[k]
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.terminated[i] = float(terminated)
        self.truncated[i]  = float(truncated)
        self.log_probs[i]  = log_prob
        self.values[i]     = value
        self.pos += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and discounted returns.

        last_value: V(s_T) from the critic for the state AFTER the last
                    stored transition (needed for bootstrapping the final
                    advantage).

        Uses self.pos (actual number of collected steps), NOT self.rollout_length,
        because the buffer may be only partially filled if the training loop
        hit total_steps mid-rollout.

        CRITICAL: only `terminated` zeroes the bootstrap, NOT truncated.
        """
        L = self.pos  # actual steps collected, may be < rollout_length
        adv = 0.0
        for t in reversed(range(L)):
            if t == L - 1:
                next_value = last_value
                # if this last step is itself a terminal state,
                # the bootstrap from last_value should be zeroed
                next_non_terminal = 1.0 - self.terminated[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.terminated[t]

            delta = (self.rewards[t]
                     + gamma * next_value * next_non_terminal
                     - self.values[t])
            adv = delta + gamma * gae_lambda * next_non_terminal * adv
            self.advantages[t] = adv

        # Target returns for the critic are computed from raw advantages:
        self.returns[:L] = self.advantages[:L] + self.values[:L]

        # Normalize advantages across the entire collected rollout for stable policy gradients:
        adv_slice = self.advantages[:L]
        self.advantages[:L] = (adv_slice - adv_slice.mean()) / (adv_slice.std() + 1e-8)

    def get_batches(self, batch_size: int, device: str = "cpu"):
        """Yield mini-batches (random permutation) as dicts of tensors.

        Uses self.pos (actual collected steps), not rollout_length, to avoid
        processing unfilled buffer slots that contain all-zero data.
        """
        L = self.pos  # actual steps collected, may be < rollout_length
        indices = np.random.permutation(L)

        for start in range(0, L, batch_size):
            idx = indices[start : start + batch_size]

            obs_batch = {
                k: torch.as_tensor(self.obs[k][idx], device=device)
                for k in OBS_KEYS
            }
            yield {
                "obs":        obs_batch,
                "actions":    torch.as_tensor(self.actions[idx],    device=device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx], device=device),
                "advantages": torch.as_tensor(self.advantages[idx], device=device),
                "returns":    torch.as_tensor(self.returns[idx],    device=device),
            }

    def reset(self) -> None:
        self.pos = 0


# ============================================================================
# PPO loss computation
# ============================================================================

def compute_ppo_loss(
    ac: GNNActorCritic,
    batch: dict,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[torch.Tensor, dict]:
    """Compute the combined PPO loss on one mini-batch.

    Returns (total_loss, info_dict) where info_dict contains individual
    loss components for logging.
    """
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    # re-evaluate the actions under the current policy
    _, new_log_probs, entropy, new_values = ac.get_action_and_value(obs, action=actions)

    # ---- clipped surrogate policy loss ----
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # ---- value function loss (simple MSE, no value clipping) ----
    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

    # ---- entropy bonus (encourages exploration) ----
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    return total_loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "approx_kl": (old_log_probs - new_log_probs).mean().item(),
    }
