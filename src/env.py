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
  
