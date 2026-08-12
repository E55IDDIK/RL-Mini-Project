# Custom environment logic (Gymnasium)
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
    will be added eventually
    """

    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config_path="configs/default.yaml"):
        '''Constructor'''
        super().__init__()

        #Loading configurations from yaml file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        env_config = config["environment"]
        #
        #instanciating the class attributes from config
        self.num_customers = env_config["num_customers"]
        self.num_nodes = env_config["num_customers"] + 1 # customers + depot
        self.vehicle_capacity = env_config["vehicle_capacity"]
        self.map_size = env_config["map_size"]
        self.demand_min = env_config["demand_min"]
        self.demand_max = env_config["demand_max"]
        self.max_steps = env_config["max_steps"]
        self.traffic_min = env_config["traffic_min"]
        self.traffic_max = env_config["traffic_max"]
        #
        # Action space : number of discrete actions = num_nodes to visit
        # node 0 = depot | nodes 1..N = customers
        self.action_space = spaces.Discrete(self.num_nodes)
        #so far we didnt apply the mask to hide illegal choices 
        #
        # Observation space : wrapper dictionary with set of keys 
        # [current_node, remaining_capacity,customer_demands,visited_status]
        # For now we keep the space simple
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
                shape=(self.num_nodes,),#demans for the number of customers
                dtype=np.float32
                ),
            "visited": spaces.MultiBinary(self.num_nodes),
            "known_traffic": spaces.Box(
                low=0.0,
                high=self.traffic_max,
                shape=(self.num_nodes, self.num_nodes),
                dtype=np.float32
                ),
            "known_traffic_mask": spaces.MultiBinary(
                (self.num_nodes, self.num_nodes)
                ),
        })
        #
        # Variables initialized later by reset()
        self.coordinates: Optional[np.ndarray] = None
        self.demands: Optional[np.ndarray] = None

        self.distance_matrix: Optional[np.ndarray] = None

        # True traffic known by the environment
        self.true_traffic: Optional[np.ndarray] = None

        # Traffic information currently available to the agent (0 for unknown)
        self.known_traffic: Optional[np.ndarray] = None

        # Indicates which traffic values are known to the agent / validity flag 
        # 0→2 = 0.0   (placeholder, because flag = unknown)
        self.known_traffic_mask: Optional[np.ndarray] = None

        self.current_node: Optional[int] = None
        self.remaining_capacity: Optional[float] = None
        self.visited: Optional[np.ndarray] = None
        self.step_count = 0



    def reset(self, *,seed=None, options=None):
        '''function to reset the agent'''
        super().reset(seed=seed)
      
        #
        # generating nodes coordinates
        self.coordinates = self.np_random.uniform(
            low=0,
            high=self.map_size,
            size=(self.num_nodes, 2)
        ).astype(np.float32)

        # We put the depot self.coordinates[0] the center of the map.
        self.coordinates[0] = np.array(
            [self.map_size / 2, self.map_size / 2],
            dtype=np.float32
        )

        # generating customer demands
        self.demands = np.zeros(
            self.num_nodes,
            dtype=np.float32
        )

        self.demands[1:] = self.np_random.integers(
            self.demand_min,
            self.demand_max + 1,
            size=self.num_customers
        )

        # calculating the distance matrix 
        self.distance_matrix = self._calculate_distances()

        # Generate the actual traffic conditions
        self.true_traffic = self._generate_traffic()

        # Initially, the agent knows nothing about road traffic
        self.known_traffic = np.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=np.float32
        )

        # Track which roads have been discovered
        self.known_traffic_mask = np.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=np.int8
        )

        # Initialize vehicle
        self.current_node = 0
        self.remaining_capacity = float(
            self.vehicle_capacity
        )

        # initializing the visited nodes array 
        self.visited = np.zeros(
            self.num_nodes,
            dtype=np.int8
        )
        if self.visited is None:
            raise RuntimeError("Visited array was not initialized.")
        # depot is considered visited.
        self.visited[0] = 1
        self.step_count = 0
        # 
        # Building observation 
        observation = self._get_observation()
    
        if self.coordinates is None or self.distance_matrix is None:
            raise RuntimeError(
                "Coordinates or distance matrix was not initialized."
            )
        info = {
            "coordinates": self.coordinates.copy(),
            "distance_matrix": self.distance_matrix.copy(),
        }

        return observation, info
    

    def _calculate_distances(self):
        '''function to calculate the distance matrix'''

        # we calculate the difference between every pair of nodes
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

    def _generate_traffic(self):
        """
        function that generates the true traffic conditions for all roads.
        Traffic values range from traffic_min to traffic_max
        Nearby roads tend to have similar traffic because
        traffic is influenced by spatial hotspots
        a hotspot is an area where roads tend to have higher congestion
        """

        if self.coordinates is None:
            raise RuntimeError("Coordinates must be generated first.")

        num_hotspots = 3
        # random locations for the hotspots
        hotspot_centers = self.np_random.uniform(
            low=0,
            high=self.map_size,
            size=(num_hotspots, 2)
        )

        # each hotspot is assigned a random strength
        hotspot_strengths = self.np_random.uniform(
            low=0.5,
            high=1.0,
            size=num_hotspots
        )

        # empty traffic matrix
        traffic = np.ones(
            (self.num_nodes, self.num_nodes),
            dtype=np.float32
        )

        # Generate traffic for each road
        for i in range(self.num_nodes):

            for j in range(i + 1, self.num_nodes):

                # Position of the middle of the road
                midpoint = (
                    self.coordinates[i]
                    + self.coordinates[j]
                ) / 2.0

                congestion = 0.0

                # calculating the influence of each hotspot
                for center, strength in zip(
                    hotspot_centers,
                    hotspot_strengths
                ):

                    distance = np.linalg.norm(
                        midpoint - center
                    )

                    influence = np.exp(
                        -(distance ** 2)
                        / (2 * (self.map_size / 4) ** 2)
                    )

                    congestion += strength * influence

                # Add a small random variation
                noise = self.np_random.normal(
                    loc=0.0,
                    scale=0.05
                )

                congestion += noise

                # Traffic is symmetric:
                # traffic(i,j) = traffic(j,i)
                traffic[i, j] = congestion
                traffic[j, i] = congestion

        # normalizing traffic values

        upper_triangle = np.triu_indices(
            self.num_nodes,
            k=1
        )

        values = traffic[upper_triangle]

        min_value = values.min()
        max_value = values.max()

        if max_value > min_value:

            normalized = (
                (traffic - min_value)
                / (max_value - min_value)
            )

        else:
            normalized = np.zeros_like(traffic)

        traffic = (
            self.traffic_min
            + normalized
            * (
                self.traffic_max
                - self.traffic_min
            )
        )

        # No travel cost from a node to itself
        np.fill_diagonal(traffic, 1.0)

        return traffic.astype(np.float32)

    
    def _reveal_traffic(self, from_node: int, to_node: int):
        """
        function to reveal the traffic on the road between two nodes
        The agent discovers the traffic as he goes 
        """
        if (
            self.true_traffic is None
            or self.known_traffic is None
            or self.known_traffic_mask is None

        ):
            raise RuntimeError(
            "Environment has not been initialized. "
            "Call reset() first."
            )

        # Get the actual traffic from the environment
        traffic = self.true_traffic[from_node, to_node]

        # Reveal it to the agent
        self.known_traffic[from_node, to_node] = traffic

        # Traffic is currently symmetric
        self.known_traffic[to_node, from_node] = traffic

        # Mark the road as known
        self.known_traffic_mask[from_node, to_node] = 1
        self.known_traffic_mask[to_node, from_node] = 1
        return float(traffic)


    def _get_observation(self):
        '''function to create observation'''

        if (
            self.current_node is None
            or self.remaining_capacity is None
            or self.demands is None
            or self.visited is None
            or self.known_traffic is None
            or self.known_traffic_mask is None
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
            "known_traffic": self.known_traffic.copy(),
            "known_traffic_mask": self.known_traffic_mask.copy(),
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
