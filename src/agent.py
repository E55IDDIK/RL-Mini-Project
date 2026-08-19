""" Implémentation des deux agents DRL comparés dans ce projet :
  - GNNDQNAgent : value-based, Double DQN + action masking.
  - GNNPPOAgent : policy-based, acteur-critique PPO + action masking. """

from __future__ import annotations
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gnn_encoder import GNNEncoder

Transition = namedtuple(
  "Transition", ["node_features", "edge_weights", "action", "reward", "next_node_features", "next_edge_weights", "next_mask", "done"]
)
#=========================================================================================================================================================================================================================================================
#GNN-DQN (Double DQN)
#=========================================================================================================================================================================================================================================================
class QNetwork(nn.Module):
  def __init__(self, node_feat_dim, hidden_dim, num_layers):
    super().__init__()
    self.encoder = GNNEncoder(node_feat_dim, hidden_dim, num_layers)
    self.q_head = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
    )
  def forward(self, node_features, edge_weights):
    h = self.encoder(node_features, edge_weights)
    return self.q_head(h).squeeze(-1)

class ReplayBuffer:
  def __init__(self, capacity):
    self.buffer = deque(maclen=capacity)

  def push(self, *args):
    self.buffer.append(Transition(*args))

  def sample(self, natch_size):
    return random.sample(self.buffer, batch_size)

  def __len__(self):
    return len(self.buffer)

class GNNDQNAgent:
  name = "GNN-DQN"

  def __init__(self, config: dict, device=None):
    self.config = config
    self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.gamma = float(config["gamma"])
    self.batch_size = int(config["batch_size"])
    self.target_update_freq = int(config["target_update_freq"])
    self.grad_clip_norm = float(config["grad_clip_norm"])

    nfd, hd, nl = config["node_feat_dim"], config["hidden_dim"], config["num_layers"]
    self.q_net = QNetwork(nfd, hd, nl).to(self.device)
    self.target_net = QNetwork(nfd, hd, nl).to(self.device)
    self.target_net.load_state_dict(self.q_net.state_dict())
    self.target_net.eval()

    


