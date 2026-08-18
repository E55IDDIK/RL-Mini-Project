# =============================================================================
# Part 1: the DVRP instance generator.
# =============================================================================
from __future__ import annotations
import numpy as np
from gymnasium import space
import yaml

def load_env_config(path: str) -> dict:
  with open(path, "r") as f:
    return yaml.safe_load(f)


class VRPGraphEnv(gym.Env):
  """ Environnement VRP mono-véhicule sur graphe cmplet avec trafic incertain """
  metadata = {"render_modes": ["human", "rgb_array"]}
  def __init__ (self, config: dict | str, seed: int | None = None):
    super().__init__()
    if isinstance(config, str):
      config = load_env_config(config)
    self.config = config

    self.num_customers = int(config["num_customers"])
    self.num_nodes = self.num_customers +1 
    self.vehicle_capacity = float(config["vehicle_capacity"])
    self.demand_range = tuple(config["demand_range"])
    self.traffic_range = tuple(config["traffic_range"])
    self.coord_range = tuple(config["coord_range"])
    self.max_steps_multiplier = int(config.get("max_steps_multiplier", 3))

    self._rng = np.random.default_rng(seed)

    self.observation_space = space.Dict(
      {
          "node_features":space.Box(
              low=-np.inf, high=np.inf, shape=(self.num_node, 5), dtype=np.float32
          ),
          "edge_weights": spaces.Box(
              low=0.0, high=np.inf, shape=(self.num_nodes, self.num_nodes), dtype=np.float32
          ),
          "action_mask": space.Box(
              low=0, high=1, shape=(self.num_nodes,), dtype=np.int8
          ),
      }
    )
    self.action_space = spaces.Discrete(self.num_nodes)
    self.reset(seed=seed)


#-------------------------------------------------------------------------------------------------------------------------------------------------------
def reset(self, *, seed: int | None= None, option: dict | None = None):
    if seed in not None:
      self._rng = np.random.default_rng(seed)

    n = self.num_nodes
    lo, hi = self.coord_range
    self.coords = self._rng.uniform(lo, hi, size=(n, 2))

    self.demands = np.zeros(n, dtype=np.float32)
    self.demands[1:] = self._rng.uniform(*self.demand_range, size=n - 1).astype(np.float32)

    diff = self.coords[:, None, :] -self.coords[None, :, :]
    base_dict = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

    mult = self._rng.uniform(*self.traffic_range, size=(n, n)).astype(np.float32)
    mult = (mult + mult.T) / 2.0
    np.fill_diagonal(mult, 0.0)
    self.edge_weights = base_dict * mult

    self.visited = np.zeros(n, dtype=bool)
    self.visited[0] = True
    self.current_node = 0
    self.remaining_capacity = self.vehicle_capacity
    self.total_cost = 0.0
    self.step_count = 0
    self.max_steps = self.max_steps_multiplier * n
    self.path = [0] #historique des noeuds visités (pour rendu/vidéo)

    obs = self._get_obs()
    info = {"coords": self.coords.copy(), "demands": self.demands.copy()}
    return obs, info
  def step(self, action: int):
    self.step_count +=1
    mask = self._action_mask()

    if mask[action] == 0:
      reward = -10.0
      terminated = True
      truncated = False
      info = {"invalid_action": True}
      return self._get_ons(), reward, terminated, truncated, info

    cost = float(self.edge_weights[self.current_node, action])
    self.total_cost += cost
    reward = -cost

    if action == 0:
      self.remaining_capacity = self.vehicle_capacity
    else:
      self.visited[action] = True
      self.remaining_capacity -= self.demands[action]

    self.current_node = action
    self.path.append(action)
    all_done = bool(self.visited[1:].all())
    terminated = all_done and action == 0
    truncated = self.step_count >= self.max_steps

    if terminated:
      reward += 1.0

    info = {"total_cost": self.total_cost, "all_customers_served": all_done}
    return self._get_obs(), reward, terminated, truncated, info

#--------------------------------------------------------------------------------------------
  def _action_mask(self) -> np.ndarray:
    mask = np.zeros(self.num_nodes, dtype=np.int8)
    unvisited = ~self.visited
    can_serve = unvisited & (self.demands <= self.remaining_capacity +1e-6)
    mask[1:] = can_serve[1:]

    all_done = bool(self.visited[1:].all())
    if all_done:
      mask[0] = 1
    else:
      mask[0] = 1 if (not can_serve[1:].any()) else 0
      if self.current_node == 0 and self.step_count == 0:
        mask[0] = 0
      if not mask.any():
        mask[0] = 1
      return mask
    def _get_obs(self):
      n = self.num_nodes
      feats = np.zeros((n, 5), dtype=np.float32)
      feats[:, 0:2] = self.coords
      feats[:, 2] = self.demands
      feats[:, 3] = self.visited.astype(np.float32)
      feats[:, 4] = 0.0
      feats[self.current_node, 4] = 1.0
      return {
          "node_features": feats,
          "edge_weights": self.edge_weights.copy(),
          "action_mask": self._action_mask(),
      }
    def render(self):
      print(
        f"Node={self.current_node} visited={self.visited.sum()}/{self.num_nodes}"
        f"cap_left={self.remaining_capacity:.2f} cost={self.total_cost:.3f}"
      )

if __name__ == "__main__":
  cfg = load_env_config("config/env.yaml")
  env = VRPGraphEnv(cfg, seed=0)
  obs, info = env.reset(seed=0)
  done = False
  total_r = 0.0
  while not done:
    mask = obs["action_mask"]
    valid = np.flatnonzero(mask)
    a = np.random.choice(valid)
    obs, r, term, trunc, info = env.step(a)
    total_r +=r
    done = term or trunc
  print("Episode finished. Total_reward =", total_r, "info =", info)
    


    
                                    
    
