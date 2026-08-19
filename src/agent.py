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
    
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(config["learning_rate"]))
    self.buffer = ReplayBuffer(int(config["buffer_capacity"]))
    self.train_steps = 0

    self.epsilon = float(config["epsilon_start"])
    self.epsilon_min = float(config["epsilon_min"])
    self.epsilon_decay = float(config["epsilon_decay"])

  def reset(self):
    pass

  def act(self, obs, explore: bool = False) -> int:
    mask = obs["action_mask"]
    valid_actions = np.flatnozero(mask)
    if explore and random.random() < self.epsilon:
      return int(np.random.choice(valid_actions))

    nf = torch.tensor(obs["node_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
    ew = torch.tensor(obs["edge_weights"], dtype=torch.float32,  device=self.device).unsqueeze(0)
    with torch.no_grad():
      q = self.q_net(nf, ew).squeeze(0), cpu().numpy()

    q_masked = np.wehre(mask.astype(bool), q, -1e9)
    return int(np.argmax(q_masked))

  def store(self, obs, acion, reward, next_obs, done):
    self.buffer.push(
      obs["node_features"], obs["edge_weight"], action, reward, next_obs["node_features"], next_obs["edge_weight"], next_obs["action_mask], done,  
    )

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

  def update(self):
    if len(self.buffer) < self.batch_size:
        return None
    batch = self.buffer.sample(self.batch_size)
    nf = torch.tensor(np.array([b.node_features for b in batch]), dtype=torch.float32, device=self.device)
    ew = torch.tensor(np.array([b.edge_weights for b in batch]), dtype=torch.float32, device=self.device)
    actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
    rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
    next_nf = torch.tensor(np.array([b.next_node_features for b in batch]), dtype=torch.float32, device=self.device)
    next_ew = torch.tensor(np.array([b.next_edge_weights for b in batch]), dtype=torch.float32, device=self.device)
    next_mask = torch.tensor(np.array([b.next_mask for b in batch]), dtype=torch.bool, device=self.device)
    dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

    q_values = self.q_net(nf, ew)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    

