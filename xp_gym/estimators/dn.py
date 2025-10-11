from flax import struct
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import haversine

from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.observation import Observation
from or_gymnax.rideshare import ManhattanRidesharePricing

@struct.dataclass
class DNEstimatorState(EstimatorState):
    """
    State for Difference-in-Networks (DN) estimator - simplified to match original DQ.
    """
    counts: jnp.ndarray  # Shape: (max_clusters,) - observations per cluster
    estimates: jnp.ndarray  # Shape: (max_clusters,) - reward estimates per cluster
    cluster_treatments: jnp.ndarray  # Shape: (max_clusters,) - treatment assignments per cluster
    design_p: float = 0.5  # Treatment probability from design


@struct.dataclass  
class DNEstimator(Estimator):
    """
    Difference-in-Networks (DN) estimator - simplified to exactly match original DQ logic.
    """
    
    # Design parameters
    lookahead_steps: int = 600  # How far back to look for temporal spillovers
    max_spatial_distance: int = 2  # Maximum spatial distance for spillovers (in km)
    switch_every: int = 5000   # Time period duration for spatiotemporal clusters
    n_events: int = 50000      # Total number of events in simulation
    n_cars: int = 300          # Total number of cars in simulation
    
    # Computed parameters (set during reset)
    _max_clusters: int = 0
    _time_ids: jnp.ndarray = None  # Time cluster for each cluster
    _space_ids: jnp.ndarray = None  # Space cluster for each cluster
    _space_adj: jnp.ndarray = None  # Spatial adjacency matrix
    
    def reset(self, rng, env_params):
        # Load spatial data to construct adjacency matrix exactly like original
        zones = pd.read_parquet("data/taxi-zones.parquet")
        unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
        zones["zone_id"] = unq_zone_ids
        nodes = pd.read_parquet("data/manhattan-nodes.parquet")
        nodes["lng"] = nodes["lng"].astype(float)
        nodes["lat"] = nodes["lat"].astype(float)
        nodes_zones = nodes.merge(zones, on="osmid")
        
        # Calculate spatial adjacency matrix like original
        centroids = nodes_zones.groupby("zone_id").aggregate({"lat": "mean", "lng": "mean"})
        n_zones = len(centroids)
        zone_dists = np.zeros((n_zones, n_zones))
        for i in range(n_zones):
            for j in range(n_zones):
                zone_dists[i, j] = haversine.haversine(
                    (centroids.iloc[i]["lat"], centroids.iloc[i]["lng"]),
                    (centroids.iloc[j]["lat"], centroids.iloc[j]["lng"]),
                )
        
        space_adj = zone_dists < self.max_spatial_distance
        
        # Calculate time and cluster structure exactly like original
        _env_params = ManhattanRidesharePricing(n_cars=self.n_cars, n_events=self.n_events).default_params
        time_ids = (_env_params.events.t // self.switch_every + 1) * self.switch_every
        unq_times, unq_time_ids = jnp.unique(time_ids, return_inverse=True)
        
        # Map event sources to zones exactly like original
        space_ids = nodes_zones.set_index(nodes_zones.index).iloc[_env_params.events.src]["zone_id"].values
        cluster_ids = unq_time_ids * n_zones + space_ids
        max_clusters = len(unq_times) * n_zones
        
        # Store precomputed values exactly like original DQ setup
        object.__setattr__(self, '_max_clusters', max_clusters)
        object.__setattr__(self, '_time_ids', jnp.repeat(unq_times, n_zones))
        object.__setattr__(self, '_space_ids', jnp.tile(jnp.arange(n_zones), len(unq_times)))
        object.__setattr__(self, '_space_adj', jnp.asarray(space_adj))
        
        return DNEstimatorState(
            counts=jnp.zeros(self._max_clusters),
            estimates=jnp.zeros(self._max_clusters),
            cluster_treatments=jnp.zeros(self._max_clusters),
            design_p=0.5,
        )

    def update(self, state: DNEstimatorState, obs: Observation):
        """
        Update DN estimator - simplified to exactly match original DQ logic.
        """
        # Extract info exactly like original ExperimentInfo structure
        cluster_id = obs.design_info.cluster_id
        treatment = obs.action.astype(jnp.float32)
        reward = obs.reward
        p = obs.design_info.p
        current_time = obs.design_info.current_time
        space_id = obs.design_info.space_cluster
        
        # Apply DQ update logic exactly like original
        z = treatment
        xi = z * (1 - p) / p + (1 - z) * p / (1 - p)
        update_val = xi * reward
        
        # Temporal adjacency exactly like original
        is_adjacent_t = (self._time_ids >= current_time - self.lookahead_steps) & (
            self._time_ids <= (current_time // self.switch_every + 1) * self.switch_every
        )
        
        # Spatial adjacency exactly like original
        is_adjacent_space = self._space_adj[self._space_ids, space_id]
        
        # Update estimates exactly like original DQ
        update_ests = state.estimates + jnp.where(
            is_adjacent_t & is_adjacent_space,
            update_val,
            0.0,
        )
        update_ests = update_ests.at[cluster_id].add(reward - update_val)
        
        # Update counts and treatments (only for main cluster)
        new_counts = state.counts.at[cluster_id].add(1)
        new_treatments = state.cluster_treatments.at[cluster_id].set(treatment)
        
        return DNEstimatorState(
            counts=new_counts,
            estimates=update_ests,
            cluster_treatments=new_treatments,
            design_p=p,
        )

    def estimate(self, state: DNEstimatorState):
        """
        Compute DN estimate - exactly matching original DQ logic.
        """
        # Apply DQ estimation exactly like original
        N = state.counts.sum()
        mask = state.counts > 0
        z = state.cluster_treatments
        p = state.design_p
        
        # Return 0 if no observations
        def compute_estimate():
            eta = z / p - (1 - z) / (1 - p)
            avg_y = (mask * state.estimates).sum() / state.counts.sum()
            baseline = state.counts * avg_y
            return (mask * eta * (state.estimates - baseline)).sum() / N
        
        return jax.lax.cond(
            N > 0,
            lambda: compute_estimate(),
            lambda: 0.0
        )