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

    


    
                                    
    
