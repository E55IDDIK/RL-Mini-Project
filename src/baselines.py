from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """Common interface every routing policy implements."""

    @abstractmethod
    def act(self,obs:dict)->int:
        """Choose the next node to visit, given only the current observation."""
        pass
    def reset(self, seed: int | None = None):
        """Called at the start of each episode."""
        pass


class RandomPolicy(Policy):

    def __init__(self, seed: int | None = None):
        self.default_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        use_seed = seed if seed is not None else self.default_seed
        self.rng = np.random.default_rng(use_seed)

    def act(self, obs: dict) -> int:
        legal = np.flatnonzero(obs["action_mask"])
        return int(self.rng.choice(legal))
    

class GreedyNearestNeighbor(Policy):

    def act(self,obs:dict)->int:
        current = int(np.argmax(obs["node_feats"][:,5])) # current position (node)
        mask = obs["action_mask"].astype(bool)
        dists = np.where(mask, obs["dist"][current], np.inf) # illegal nodes can't be picked

        return int(np.argmin(dists))
