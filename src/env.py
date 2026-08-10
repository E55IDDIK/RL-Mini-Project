# Custom environment logic (Gymnasium)

#generating the routing instance
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from typing import Optional


class DynamicVRPEnv(gym.Env):
    """
    Dynamic Vehicle Routing Environment.

    Version 1:
    - One depot
    - Multiple customers
    - One vehicle
    - Customer demands
    - Vehicle capacity
    - Static Euclidean distance matrix

    Traffic, partial observability, rewards, and action masking
    will be added in later versions.
    """

    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config_path="configs/default.yaml"):
        '''Constructor'''
        super().__init__()

        #Loading configurations
        #opening the config file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        env_config = config["environment"]

        self.num_customers = env_config["num_customers"]
        self.num_nodes = self.num_customers + 1 #+depot

        self.vehicle_capacity = env_config["vehicle_capacity"]

        self.map_size = env_config["map_size"]

        self.demand_min = env_config["demand_min"]
        self.demand_max = env_config["demand_max"]

        self.max_steps = env_config["max_steps"]


        # ---------------------------------------------------------
        # Action space : num_nodes discrete actions (num_nodes to visit)
        # Node 0 = depot
        # Nodes 1..N = customers
        self.action_space = spaces.Discrete(self.num_nodes)
        #so far we didnt apply the mask to hide illegal choices 

        # ---------------------------------------------------------
        # Observation space 
        # For now we keep the observation simple: 
        # [current_node,
        #  remaining_capacity,
        #  customer_demands,
        #  visited_status]
        #
        self.observation_space = spaces.Dict({
            "current_node": spaces.Discrete(self.num_nodes),
            "remaining_capacity": spaces.Box(
                low=0,
                high=self.vehicle_capacity,
                shape=(1,),
                dtype=np.float32
            ),
            "demands": spaces.Box(
                low=0,
                high=self.demand_max,
                shape=(self.num_nodes,),
                dtype=np.float32
            ),
            "visited": spaces.MultiBinary(self.num_nodes),
        })

        # ---------------------------------------------------------
        # Variables initialized later by reset()

        self.coordinates: Optional[np.ndarray] = None
        self.demands: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None

        self.current_node: Optional[int] = None
        self.remaining_capacity: Optional[float] = None
        self.visited: Optional[np.ndarray] = None

        self.step_count = 0

    # ***********************************************************

    def reset(self, *,seed=None, options=None):

        super().reset(seed=seed)

        # ---------------------------------------------------------
        # Generate node coordinates
        # ---------------------------------------------------------

        self.coordinates = self.np_random.uniform(
            low=0,
            high=self.map_size,
            size=(self.num_nodes, 2)
        ).astype(np.float32)

        # ---------------------------------------------------------
        # Depot is node 0
        # ---------------------------------------------------------

        # We put the depot at the center of the map.
        self.coordinates[0] = np.array(
            [self.map_size / 2, self.map_size / 2],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Generate customer demands
        # ---------------------------------------------------------

        self.demands = np.zeros(
            self.num_nodes,
            dtype=np.float32
        )

        self.demands[1:] = self.np_random.integers(
            self.demand_min,
            self.demand_max + 1,
            size=self.num_customers
        )

        # ---------------------------------------------------------
        # Calculate pairwise distances
        # ---------------------------------------------------------

        self.distance_matrix = self._calculate_distances()

        # ---------------------------------------------------------
        # Initialize vehicle
        # ---------------------------------------------------------

        self.current_node = 0

        self.remaining_capacity = float(
            self.vehicle_capacity
        )

        # ---------------------------------------------------------
        # Visited array
        # ---------------------------------------------------------

        self.visited = np.zeros(
            self.num_nodes,
            dtype=np.int8
        )

        # Depot is considered visited.
        self.visited[0] = 1

        self.step_count = 0

        # ---------------------------------------------------------
        # Build observation
        # ---------------------------------------------------------

        observation = self._get_observation()

        info = {
            "coordinates": self.coordinates.copy(),
            "distance_matrix": self.distance_matrix.copy(),
        }

        return observation, info

    # =============================================================
    # DISTANCE MATRIX
    # =============================================================

    def _calculate_distances(self):

        # Difference between every pair of nodes
        coordinates = np.asarray(self.coordinates)

        differences = (
            coordinates[:, np.newaxis, :]
            - coordinates[np.newaxis, :, :]
        )

        # Euclidean distance
        distances = np.sqrt(
            np.sum(differences ** 2, axis=2)
        )

        return distances.astype(np.float32)

    # =============================================================
    # OBSERVATION
    # =============================================================

    def _get_observation(self):

        if (
            self.current_node is None
            or self.remaining_capacity is None
            or self.demands is None
            or self.visited is None
        ):
            raise RuntimeError(
                "Environment has not been initialized. "
                "Call reset() first."
            )

        observation = {
            "current_node": self.current_node,

            "remaining_capacity": np.array(
                [self.remaining_capacity],
                dtype=np.float32
            ),

            "demands": self.demands.copy(),

            "visited": self.visited.copy(),
        }

        return observation

    # =============================================================
    # STEP
    # =============================================================

    def step(self, action):

        # This will be implemented next.
        raise NotImplementedError(
            "step() will be implemented in the next stage."
        )
