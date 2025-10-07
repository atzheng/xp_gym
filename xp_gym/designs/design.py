import operator as op
from typing import Tuple

from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import EnvParams, EnvState
import jax
import jax.numpy as jnp
from numpy import searchsorted
import pandas as pd
import numpy as np
from haversine import haversine

from xp_gym.observation import Observation


@struct.dataclass
class DesignInfo:
    """
    Information about the design at a given time step, which is ultimately
    wrapped into the Observation and passed to the Estimator.
    """
    p: float  # Probability of treatment assignment


@struct.dataclass
class DesignState:
    """
    Dynamic state of the experimental design. Stores nothing by default.
    """
    pass


@struct.dataclass
class Design:
    """
    Design class that holds the parameters of the experimental design.
    """
    def reset(self, rng: PRNGKey, env_params: EnvParams) -> DesignState:
        return DesignState()

    def update(self, state: DesignState, obs: Observation):
        return state

    def assign_treatment(
        self, design_state: DesignState, env_state: EnvState
    ) -> Tuple[bool, DesignInfo]:
        raise NotImplementedError()


@struct.dataclass
class ClusterDesignState(DesignState):
    # A single seed that determines cluster assignments throughout.
    rng: PRNGKey


@struct.dataclass
class ClusterDesignInfo:
    p: float  # Probability of treatment assignment at time t
    cluster_id: int  # Cluster ID for the observation at time t


@struct.dataclass
class ClusterRandomizedDesign(Design):
    """
    Base class for designs that assign clusters based on a treatment.
    """
    p: float

    def reset(self, rng: PRNGKey, env_params: EnvParams) -> ClusterDesignState:
        return ClusterDesignState(rng=rng)

    def get_cluster_id(self, design_state, env_state) -> int:
        """
        Returns the cluster ID for the current state.
        This method should be implemented by subclasses to return the appropriate cluster ID.
        """
        raise NotImplementedError()

    def assign_treatment(self, design_state: ClusterDesignState, env_state: EnvState):
        cluster_id = self.get_cluster_id(design_state, env_state)
        new_rng = jax.random.fold_in(design_state.rng, cluster_id)
        z = jax.random.bernoulli(new_rng, self.p)
        return (
            z,
            ClusterDesignInfo(p=self.p, cluster_id=cluster_id),
        )

@struct.dataclass
class UnitRandomizedDesign(ClusterRandomizedDesign):
    def get_cluster_id(self, _, env_state) -> bool:
        # Unique cluster at every time step
        return env_state.time


@struct.dataclass
class SwitchbackDesign(ClusterRandomizedDesign):
    time_attr: str = "time"
    frequency: float = 1.0

    def get_cluster_id(self, design_state: ClusterDesignState, env_state: EnvState) -> int:
        """
        Returns the cluster ID based on the time attribute.
        This design uses the time step as the cluster ID.
        """
        t = op.attrgetter(self.time_attr)(env_state)
        return jnp.floor(t / self.frequency).astype(jnp.int32)

@struct.dataclass
class SpatioClusterDesignState(ClusterDesignState):
    """Simplified state for ascending time-ordered events."""
    current_time_index: int  # Current sequential time period index
    last_time_interval: int  # Last seen time interval boundary


@struct.dataclass
class SpatioClusterDesignInfo(ClusterDesignInfo):
    """Extended design info that includes spatial mapping information for DN estimator."""
    time_cluster: int  # Time-based cluster component
    space_cluster: int  # Space-based cluster component
    n_spatial_zones: int  # Total number of spatial zones
    zone_distances: jnp.ndarray  # Distance matrix between zones (km)
    switch_every: int  # Time period duration from design
    current_time: int  # Current event time for temporal adjacency calculations
    src_2_zone: jnp.ndarray  # Mapping from src location index to zone_id
    nodes_zones_available: bool = True  # Whether real spatial mapping is available
    env_state: EnvState = None  # Environment state for temporal adjacency calculations


@struct.dataclass
class SpatioClusterDesign(ClusterRandomizedDesign):
    """
    Efficient spatiotemporal design leveraging ascending time order.
    
    Key improvements over SpatiotemporalDesign:
    1. No dynamic arrays or searching - just track current time index
    2. Simple increment when time period changes  
    3. No ad-hoc size limits
    4. Fully JAX-compatible operations with O(1) complexity
    """
    switch_every: int = 5000
    data_path: str = "."
    _spatial_mapping: dict = None

    def __post_init__(self):
        """Load spatial data during initialization."""
        if self._spatial_mapping is None:
            try:
                # Load spatial data exactly like original SpatiotemporalDesign
                zones = pd.read_parquet(f"{self.data_path}/taxi-zones.parquet")
                unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
                zones["zone_id"] = unq_zone_ids
                nodes = pd.read_parquet(f"{self.data_path}/manhattan-nodes.parquet")
                nodes["lng"] = nodes["lng"].astype(float)
                nodes["lat"] = nodes["lat"].astype(float)
                nodes_zones = nodes.merge(zones, on="osmid")
                
                # Compute zone distance matrix
                centroids = nodes_zones.groupby("zone_id").aggregate(
                    {"lat": "mean", "lng": "mean"}
                )
                n_zones = len(centroids)
                zone_distances = np.zeros((n_zones, n_zones))
                
                for i in range(n_zones):
                    for j in range(n_zones):
                        zone_distances[i, j] = haversine(
                            (centroids.iloc[i]["lat"], centroids.iloc[i]["lng"]),
                            (centroids.iloc[j]["lat"], centroids.iloc[j]["lng"]),
                        )
                
                # Create JAX-compatible spatial mapping array
                max_src_idx = nodes_zones.index.max()
                src_to_zone = np.full(max_src_idx + 1, -1, dtype=np.int32)
                for idx, row in nodes_zones.iterrows():
                    src_to_zone[idx] = row["zone_id"]
                
                # Store as JAX arrays
                object.__setattr__(self, '_spatial_mapping', {
                    'nodes_zones': nodes_zones,
                    'n_zones': n_zones,
                    'zone_distances': jnp.asarray(zone_distances),
                    'src_to_zone': jnp.asarray(src_to_zone)
                })
                
                print(f"RefinedSpatioClusterDesign: Loaded {len(nodes_zones)} nodes, {n_zones} zones")
                
            except Exception as e:
                print(f"Error loading spatial data: {e}")
                raise

    def reset(self, rng: PRNGKey, env_params: EnvParams) -> SpatioClusterDesignState:
        """Simple reset with no ad-hoc parameters."""
        return SpatioClusterDesignState(
            rng=rng,
            current_time_index=-1,  # Start with -1, will increment to 0 on first use
            last_time_interval=-1  # No time interval seen yet
        )

    def get_cluster_id(self, design_state: SpatioClusterDesignState, env_state: EnvState) -> int:
        """
        Efficient cluster ID calculation leveraging ascending time order.
        
        Key insight: Since time is ascending, we only need to check if we've 
        moved to a new time period. If so, increment the time index.
        """
        # Calculate current time interval (end boundary) - same as original design
        t = env_state.event.t
        current_time_interval = (t // self.switch_every + 1) * self.switch_every
        
        # For first call or new time period, we need to increment
        # This matches the original design's sequential assignment behavior
        is_new_period = current_time_interval != design_state.last_time_interval
        
        # Calculate time index - start from 0 for first period
        time_index = jnp.where(
            is_new_period,
            design_state.current_time_index + 1,  # Increment for new period
            design_state.current_time_index       # Keep current index
        )
        
        # Get space cluster using JAX array lookup
        src_location = env_state.event.src
        space_cluster = self._spatial_mapping['src_to_zone'][src_location]
        
        # Combine into cluster ID - same formula as original design
        n_zones = self._spatial_mapping['n_zones']
        cluster_id = time_index * n_zones + space_cluster
        
        return cluster_id

    def assign_treatment(self, design_state: SpatioClusterDesignState, env_state: EnvState):
        """Assign treatment with efficient O(1) cluster ID calculation."""
        # Get cluster ID
        cluster_id = self.get_cluster_id(design_state, env_state)
        
        # Generate treatment assignment
        new_rng = jax.random.fold_in(design_state.rng, cluster_id)
        z = jax.random.bernoulli(new_rng, self.p)
        
        # Extract components for design info
        n_zones = self._spatial_mapping['n_zones']
        time_cluster = cluster_id // n_zones
        space_cluster = cluster_id % n_zones
        
        # Get current time for design info
        current_time = env_state.event.t
        
        design_info = SpatioClusterDesignInfo(
            p=self.p,
            cluster_id=cluster_id,
            time_cluster=time_cluster,
            space_cluster=space_cluster,
            n_spatial_zones=n_zones,
            zone_distances=self._spatial_mapping['zone_distances'],
            switch_every=self.switch_every,
            current_time=current_time,
            src_2_zone=self._spatial_mapping['src_to_zone'],
            nodes_zones_available=True,
            env_state=env_state
        )
        
        return z, design_info

    def update(self, state: SpatioClusterDesignState, obs: Observation) -> SpatioClusterDesignState:
        """
        Update design state based on the observation.
        This is called after assign_treatment in the simulation loop.
        """
        # Extract current time from the design info in the observation
        if hasattr(obs.design_info, 'current_time'):
            current_time = obs.design_info.current_time
            current_time_interval = (current_time // self.switch_every + 1) * self.switch_every
            
            # Check if we've moved to a new time period
            is_new_period = current_time_interval != state.last_time_interval
            
            # Update state if needed (leveraging ascending time order)
            new_time_index = jnp.where(
                is_new_period,
                state.current_time_index + 1,  # Increment for new period
                state.current_time_index       # Keep current index
            )
            
            new_last_interval = jnp.where(
                is_new_period,
                current_time_interval,         # Update to current interval
                state.last_time_interval      # Keep previous
            )
            
            return state.replace(
                current_time_index=new_time_index,
                last_time_interval=new_last_interval
            )
        
        return state