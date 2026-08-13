# =============================================================================
# Part 1: the DVRP instance generator.
# =============================================================================
from dataclasses import dataclass
import numpy as np
import yaml

def load_config(path: str)-> dict:
    """Load a YAML config file into a plain nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
@dataclass
class DVRPInstance:
    coords: np.ndarray
    demands: np.ndarray
    dist: np.ndarray
    traffic: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray

def _congestion_field(points,centers,strengths,scale):
    d1 = np.abs(points[:, None, :] - centers[None, :, :]).sum(-1)   # Manhattan distance to each center
    d2 = d1 ** 2
    return (strengths[None, :] * np.exp(-d2 / (2.0 * scale ** 2))).sum(-1)

def generate_instance(cfg: dict,rng:np.random.Generator)-> DVRPInstance:
    prob = cfg["problem"]
    traf = cfg["traffic"]
    tw = cfg["time_windows"]
    map_size = float(cfg["map"]["size"])
    N = prob["n_customers"] + 1  # + depot


    #position : depot fixed at center, customers random in [0,map_size]^2
    coords = rng.random((N,2)).astype(np.float32)*map_size
    coords[0] = np.array([map_size/2, map_size/2],dtype=np.float32)
    #demands: random per customer, 0 for the depot
    demands = rng.integers(prob["demand_low"],prob["demand_high"] + 1,size=N).astype(np.float32)
    demands[0] = 0.0

    #distance : Manhattan calculate the distance between each node and every other node

    diff = coords[:,None,:] - coords[None,:,:]
    dist = np.abs(diff).sum(-1).astype(np.float32)

    #traffic: spatially-correlated congestion field, mapped to [low, high]
    n_centers = int(rng.integers(2,5))
    centers = rng.random((n_centers,2)).astype(np.float32)*map_size
    strengths = rng.uniform(0.5,1.0,size=n_centers).astype(np.float32)
    scale = float(traf["spatial_scale"]) #0.3

    mids = (coords[None,:,:] + coords[:,None,:]) / 2.0
    field = _congestion_field(mids.reshape(-1,2),centers,strengths,scale).reshape(N,N)
    fmin,fmax = float(field.min()),float(field.max())
    b = (traf["high"] - traf["low"])/(fmax-fmin + 1e-8) # 1e-8: tiny number to avoid division by zero
    a = traf["low"] - b * fmin
    traffic = (a + b * field).astype(np.float32)
    traffic = 0.5 * (traffic + traffic.T)  # make symmetric
    np.fill_diagonal(traffic, 1.0)  # no self-congestion

    # time windows: random per customer

    window_start = np.zeros(N, dtype=np.float32)
    window_end = np.full(N,np.inf,dtype=np.float32)

    if tw["enabled"]:
        starts = rng.uniform(tw["earliest_start"],tw["latest_start"],size=N-1).astype(np.float32)
        widths = rng.uniform(tw["window_width_low"],tw["window_width_high"],size=N-1)
        window_start[1:] = starts.astype(np.float32)
        window_end[1:] = (starts + widths).astype(np.float32)

    return DVRPInstance(coords,demands,dist,traffic,window_start,window_end)

# =============================================================================
# Part 2: the Gymnasium environment itself.
# =============================================================================

import gymnasium as gym
from gymnasium import spaces

# node feature layout, one row per node:
# [x, y, remaining_demand, served_flag, is_depot, is_current, window_start, window_end]
NODE_FEAT_DIM = 8

class DynamicCVRPEnv(gym.Env):
    """Gymnasium environment for the dynamic capacitated vehicle routing problem (DVRP)."""
    metadata = {"render_modes":[]}

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        prob = cfg["problem"]
        assert prob["demand_high"] <= prob["vehicle_capacity"], \
            "a single full load must be able to serve any one customer"
        self.N = prob["n_customers"] + 1
        tw = cfg["time_windows"]
        self._tw_horizon = float(tw["latest_start"] + tw["window_width_high"] + 1e-8)
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Dict({
            "node_feats": spaces.Box(-np.inf,np.inf,(self.N,NODE_FEAT_DIM),np.float32),
            "dist": spaces.Box(0.0,np.inf,(self.N,self.N),np.float32),
            "traffic_obs": spaces.Box(0.0,np.inf,(self.N,self.N),np.float32),
            "traffic_mask": spaces.Box(0.0,1.0,(self.N,self.N),np.float32),
            "vehicle":spaces.Box(-np.inf,np.inf,(3,),np.float32),
            "action_mask":spaces.Box(0.0,1.0,(self.N,),np.float32),
        })
        self.inst = None

        
    # ------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.inst = generate_instance(self.cfg, self.np_random)

        self.served = np.zeros(self.N, dtype=bool)          # served[0] unused (depot)
        self.current = 0                                     # vehicle starts at the depot
        self.remaining_cap = float(self.cfg["problem"]["vehicle_capacity"])
        self.steps = 0                                        # action-count safety counter
        self.elapsed_time = 0.0                               # real clock: accumulated distance x traffic
        self.total_cost = 0.0
        self.route = [0]
        self.observed = np.zeros((self.N, self.N), dtype=bool)  # which edges' traffic is revealed
        self._reveal(self.current)
        self.n_on_time = 0.0
        self.n_waited = 0.0
        self.n_late = 0.0

        return self._obs(), self._info()

    # ---------------------------------------------------------------- helpers
    def _reveal(self, node):
        """Mark traffic as observed on every edge within observation_radius of `node`."""
        radius = self.cfg["traffic"]["observation_radius"]
        near = self.inst.dist[node] <= radius
        self.observed[node, near] = True
        self.observed[near, node] = True
    
    def _valid_mask(self):
        """Boolean [N]: which nodes the vehicle may legally move to right now."""
        mask = (~self.served) & (self.inst.demands <= self.remaining_cap + 1e-9)
        mask[0] = False                 # depot excluded by default...
        if self.current != 0:
            mask[0] = True              # ...unless the vehicle isn't already there (reload option)
        mask[self.current] = False      # cannot "move" to where it already is
        return mask

    def _obs(self):
        nf = np.zeros((self.N, NODE_FEAT_DIM), dtype=np.float32)
        nf[:, 0:2] = self.inst.coords
        nf[:, 2] = np.where(self.served, 0.0, self.inst.demands)   # remaining demand
        nf[:, 3] = self.served.astype(np.float32)
        nf[0, 4] = 1.0                                             # is_depot
        nf[self.current, 5] = 1.0                                  # is_current
        nf[:, 6] = self.inst.window_start / self._tw_horizon
        # depot's window_end is +inf (no real window) -- use -1.0 as a finite
        # "no window" sentinel so this stays a well-behaved network input
        nf[:, 7] = np.where(np.isinf(self.inst.window_end), -1.0, self.inst.window_end / self._tw_horizon)

        cap = self.cfg["problem"]["vehicle_capacity"]
        max_steps = self.cfg["episode"]["max_steps"]

        return {
            "node_feats":   nf,
            "dist":         self.inst.dist,
            "traffic_obs":  np.where(self.observed, self.inst.traffic, 0.0).astype(np.float32),
            "traffic_mask": self.observed.astype(np.float32),
            "vehicle":      np.array([self.remaining_cap / cap,
                                      self.steps / max_steps,
                                      self.elapsed_time / max_steps], np.float32),
            "action_mask":  self._valid_mask().astype(np.float32),
        }
# ------------------------------------------------------------------- step
    def step(self, action):
        action = int(action)
        mask = self._valid_mask()
        if not mask[action]:
            raise ValueError(f"action {action} is masked / invalid")

        prev = self.current
        distance = float(self.inst.dist[prev, action])
        multiplier = float(self.inst.traffic[prev, action])
        cost = distance * multiplier                      # the real cost of this move
        self.total_cost += cost

        reward = -self.cfg["reward"]["w_distance"] * cost

        # --- time: when would the vehicle arrive, and does it have to wait? ---
        arrival_time = self.elapsed_time + cost
        w_start = float(self.inst.window_start[action])
        w_end = float(self.inst.window_end[action])
        service_start = max(arrival_time, w_start)          # force-to-wait if early
        lateness = max(0.0, service_start - w_end)           # >0 only if genuinely late

        if lateness > 0.0:
            reward -= self.cfg["reward"]["w_time_window_penalty"]

        # track delivery-outcome stats (customers only -- the depot's [0, inf]
        # "window" always falls into the on-time case and isn't a real delivery)
        if action != 0:
            if lateness > 0.0:
                self.n_late += 1
            elif arrival_time < w_start:
                self.n_waited += 1
            else:
                self.n_on_time += 1

        self.elapsed_time = service_start                    # the wait carries forward

        # --- move the vehicle, update capacity / served status ---
        self.current = action
        self.route.append(action)
        self.steps += 1

        if action == 0:
            self.remaining_cap = float(self.cfg["problem"]["vehicle_capacity"])   # reload
        else:
            self.served[action] = True
            self.remaining_cap -= float(self.inst.demands[action])
            reward += self.cfg["reward"]["w_delivery_bonus"]

        self._reveal(self.current)

        # --- termination / truncation ---
        all_served = bool(self.served[1:].all())
        terminated = all_served and self.current == 0
        if terminated:
            reward += self.cfg["reward"]["w_complete_bonus"]

        truncated = (self.steps >= self.cfg["episode"]["max_steps"]) and not terminated
        if truncated:
            n_unserved = int((~self.served[1:]).sum())
            reward -= self.cfg["reward"]["w_unserved_penalty"] * n_unserved

        # Safety check: a mask with zero legal actions is normal and expected
        # at a successful termination (nowhere left to go) -- but if it
        # happens *without* terminating or truncating, that's a genuine
        # dead end (a bug), not a valid outcome, so we raise loudly.
        if not terminated and not truncated and not self._valid_mask().any():
            raise RuntimeError(
                "No valid action remains, but the episode neither terminated "
                "nor truncated -- the environment reached an unexpected dead end."
            )

        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info()
        
    def _info(self):
        return {
            "route": list(self.route),
            "total_cost": self.total_cost,
            "elapsed_time": self.elapsed_time,
            "n_served": int(self.served[1:].sum()),
            "n_customers": self.cfg["problem"]["n_customers"],
            "action_mask": self._valid_mask(),
        }