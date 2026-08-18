"""GNN encoder + Q-network for the Dynamic CVRP environment.

Design notes (read this before touching the architecture):

1. No cross-step recurrent hidden state.
   The original plan included a GRU carrying a "belief state" across steps
   within an episode. That conflicts with how the training loop actually
   works: `replay_buffer.py` samples random, unordered transitions from
   across many different episodes/timesteps, and `train_dqn.py` calls
   `network(obs_batch) -> Q` with no hidden state in or out. A GRU cannot
   meaningfully carry memory between two transitions that have no temporal
   relationship to each other.

   This isn't actually a problem for us: `env.py`'s `self.observed` mask
   never resets during an episode -- it only grows -- so every observation's
   `traffic_mask` / `traffic_obs` already reflects *everything* the vehicle
   has seen so far this episode. The belief state the GRU was meant to build
   is already externalized into the environment's persistent state. So the
   network below is a pure, stateless function of the current (already
   cumulative) observation: `network(obs) -> Q-values`. This is a deliberate,
   documented divergence from the original GRU plan (see notebook.md).

2. Batching is real, not optional.
   `replay_buffer.sample()` returns dense tensors with a genuine leading
   batch dimension (B, N, N) for dist/traffic_obs/traffic_mask. The graph
   size N is fixed for the whole training run (same config), so instead of
   PyTorch Geometric's block-diagonal Batch machinery, we just loop over the
   batch dimension and run one graph forward per sample. This is simple and
   correct; it can be replaced with a fully vectorized version later if
   training speed becomes a bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Must match env.py's NODE_FEAT_DIM and the edge feature layout used
# throughout the project.
# node_feats columns: [x, y, remaining_demand, served_flag, is_depot,
#                       is_current, window_start, window_end]
NODE_FEAT_DIM = 8
IS_CURRENT_COL = 5  # index of the "is_current" one-hot column in node_feats

# edge_attr columns: [distance, observed_traffic, is_observed]
EDGE_FEAT_DIM = 3


def build_complete_edge_index(n_nodes: int) -> torch.Tensor:
    """All N*(N-1) ordered node pairs (i, j), i != j.

    Returns shape (2, N*(N-1)): row 0 = source nodes, row 1 = target nodes.
    Call this once per fixed N (e.g. in a network's __init__) and reuse it
    every forward call -- it never changes within a training run.
    """
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def build_edge_attr(
    dist: torch.Tensor,
    traffic_obs: torch.Tensor,
    traffic_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Look up [distance, observed_traffic, is_observed] for every edge in
    edge_index, from the dense (N,N) matrices.

    dist/traffic_obs/traffic_mask: (N,N), single graph (no batch dim -- the
    caller loops over the batch and calls this once per sample).
    edge_index: (2,E).
    Returns (E, 3).
    """
    src, dst = edge_index[0], edge_index[1]
    d = dist[src, dst]
    t = traffic_obs[src, dst]
    m = traffic_mask[src, dst]
    return torch.stack([d, t, m], dim=-1)


class GNNLayer(nn.Module):
    """One edge-conditioned graph convolution (NNConv / ECC).

    For each edge, a small neural net (`edge_net`) maps that edge's
    [distance, traffic, observed] features into a full (in_dim x out_dim)
    weight matrix -- so the message passed along each edge is conditioned on
    that edge's own distance/traffic, not a single shared weight for every
    edge like a plain GCN layer would use. This is what lets the network
    learn that congested edges should be treated differently from
    free-flowing ones.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        edge_hidden: int = 32,
    ):
        super().__init__()
        # Lazy import so this module doesn't hard-require torch_geometric
        # unless the real network is actually used (StubQNetwork in dqn.py
        # doesn't need it).
        from torch_geometric.nn import NNConv

        edge_net = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, in_dim * out_dim),
        )
        self.conv = NNConv(in_dim, out_dim, edge_net, aggr="mean")

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class GNNEncoder(nn.Module):
    """Two-layer edge-conditioned GNN over a single graph.

    Returns BOTH:
      - graph_summary: one fixed-size vector per observation
        [current-node embedding | global mean-pooled embedding | vehicle
        state]. Kept for any future pooled/global use (e.g. logging,
        auxiliary losses, or a future recurrent extension).
      - x: the raw per-node embeddings (N, hidden_dim). This is what the
        Q-head actually needs to score each node individually -- do not
        discard this.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 32,
        edge_feat_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        self.layer1 = GNNLayer(node_feat_dim, hidden_dim, edge_feat_dim)
        self.layer2 = GNNLayer(hidden_dim, hidden_dim, edge_feat_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.summary_dim = hidden_dim * 2 + 3  # current + global + vehicle(3)

    def forward(self, node_feats, edge_index, edge_attr, vehicle):
        """node_feats: (N, node_feat_dim). edge_attr: (E, edge_feat_dim).
        vehicle: (3,). is_current is read directly from node_feats to avoid
        passing the same information twice.
        """
        x = self.norm1(torch.relu(self.layer1(node_feats, edge_index, edge_attr)))
        x = self.norm2(torch.relu(self.layer2(x, edge_index, edge_attr)))  # (N, hidden_dim)

        is_current = node_feats[:, IS_CURRENT_COL]
        current_idx = is_current.argmax()
        current_emb = x[current_idx]
        global_emb = x.mean(dim=0)

        graph_summary = torch.cat([current_emb, global_emb, vehicle], dim=-1)
        return graph_summary, x


class GNNQNetwork(nn.Module):
    """The real GNN-DQN network: obs_batch -> Q-values, shape (B, N).

    This is what `train_dqn.py`'s `build_network(kind="real")` should return.
    It satisfies the exact same contract as `StubQNetwork` in dqn.py, so no
    other file needs to change at integration time:

        q = network(obs_batch)   # obs_batch values have a leading batch dim
        # q.shape == (B, N)
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

        # Built once, reused every forward call -- the graph is complete and
        # fixed-size for the whole training run.
        self.register_buffer("edge_index", build_complete_edge_index(n_nodes))

        # Each node is scored using its own embedding concatenated with the
        # global context (current-node embedding + mean-pooled graph + vehicle
        # state), so the Q-head has full information about remaining capacity,
        # elapsed time, and the current node's identity.
        head_input_dim = hidden_dim + self.encoder.summary_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, obs_batch: dict) -> torch.Tensor:
        node_feats = obs_batch["node_feats"]   # (B, N, F)
        dist = obs_batch["dist"]               # (B, N, N)
        traffic_obs = obs_batch["traffic_obs"]  # (B, N, N)
        traffic_mask = obs_batch["traffic_mask"]  # (B, N, N)
        vehicle = obs_batch["vehicle"]         # (B, 3)

        batch_size = node_feats.shape[0]
        edge_index = self.edge_index

        q_values = []
        for b in range(batch_size):
            edge_attr = build_edge_attr(
                dist[b], traffic_obs[b], traffic_mask[b], edge_index
            )
            graph_summary, x = self.encoder(node_feats[b], edge_index, edge_attr, vehicle[b])
            # Broadcast the global context to every node so the Q-head can
            # condition each node's score on vehicle state (capacity, time)
            # and the current-node / global-graph embeddings.
            context = graph_summary.unsqueeze(0).expand(self.n_nodes, -1)  # (N, summary_dim)
            x_aug = torch.cat([x, context], dim=-1)  # (N, hidden_dim + summary_dim)
            q_values.append(self.head(x_aug).squeeze(-1))  # (N,)

        return torch.stack(q_values, dim=0)  # (B, N)
