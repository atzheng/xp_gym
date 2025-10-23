from flax import struct
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import haversine
from dataclasses import field

from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.estimators.network import (
    InterferenceNetwork,
    SpatioTemporalInterferenceNetwork,
    SpatioTemporalInterferenceNetworkState,
)
from xp_gym.observation import Observation
from gymnax.environments.environment import Environment, EnvParams
from or_gymnax.rideshare import ManhattanRidesharePricing, obs_to_state
from jaxtyping import Array, Bool, Float, Integer
from jax import Array


def haversine_km(lon1, lat1, lon2, lat2):
    """Approximate great-circle distance (in km) between two lon/lat pairs."""
    R = 6371.0  # Earth's radius in km
    lon1, lat1, lon2, lat2 = [jnp.radians(x) for x in (lon1, lat1, lon2, lat2)]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        jnp.sin(dlat / 2) ** 2
        + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2) ** 2
    )
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    return R * c


@struct.dataclass
class SpatioTemporalNetworkInfo:
    time: Float[Array, ()] = field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float32)
    )
    location: Integer[Array, ()] = field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int32)
    )


@struct.dataclass
class RideshareTemporalNetwork(SpatioTemporalInterferenceNetwork):
    """
    Concrete implementation of spatiotemporal interference network for Manhattan.
    """

    lookahead_steps: int = 600
    max_spatial_distance: int = 2  # km
    network_info_cls = SpatioTemporalNetworkInfo

    def get_cluster_info(
        self, env: Environment, env_params: EnvParams, obs: Observation
    ) -> ClusterInfo:
        """Extract cluster info (lat, lng, t) from observation."""
        state = obs_to_state(env, env_params, obs)
        return ClusterInfo(state.event.t, lat=loc[0], lng=loc[1], t=t)

    def is_adjacent(
        self,
        env: Environment,
        env_params: EnvParams,
        x: SpatioTemporalNetworkInfo,
        y: SpatioTemporalNetworkInfo,
    ):
        """Check if the edge (x, y) exists in the interference graph (i.e., x affects y)."""
        is_space_adj = (
            env_params.distance[x.location, y.location]
            <= self.max_spatial_distance
        )
        is_time_adj = y.t - x.t < self.lookahead_steps & y.t > x.t
        return is_time_adj & is_space_adj


@struct.dataclass
class DNV2EstimatorState(EstimatorState):
    """State for DN v2 estimator using network abstractions."""

    design_cluster_treatments: Bool[Array, "n_design_cluster_ids"]
    design_cluster_treatment_probs: Float[Array, "n_design_cluster_ids"]
    design_cluster_counts: Integer[Array, "n_design_cluster_ids"]
    estimates: Float[Array, "n_design_cluster_ids"]
    design_cluster_ids: Integer[Array, "window_size"]
    network_infos: NetworkInfo
    window_ptr: int


@struct.dataclass
class DNV2Estimator(Estimator):
    """
    A correct implementation of the DN estimator for clustering.
    This makes the following approximations:

    - State discretization: `network` will take an obs and return a discrete cluster ID.
       Adjacency is defined based on these cluster ID
    - Limited memory window: The fully "correct" DN estimator would need to store all past
       observations to compute interference effects. Here, we limit to a fixed window size.
       This approximation can be removed by setting the window size tobe sufficiently large
    """

    network: InterferenceNetwork
    window_size: int
    n_design_clusters: int

    def reset(self, rng, env, env_params):
        """Initialize DN v2 estimator with network state."""
        network_infos = jax.vmap(lambda x: self.network.network_info_cls())(
            jnp.zeros((self.window_size,))  # Dummy initialization
        )

        return DNV2EstimatorState(
            design_cluster_treatments=jnp.zeros(
                (self.n_design_clusters,), dtype=jnp.bool_
            ),
            design_cluster_treatment_probs=jnp.zeros(
                (self.n_design_clusters,), dtype=jnp.float32
            ),
            design_cluster_counts=jnp.zeros(
                (self.n_design_clusters,), dtype=jnp.int32
            ),
            estimates=jnp.zeros((self.n_design_clusters,), dtype=jnp.float32),
            design_cluster_ids=jnp.zeros((self.window_size,), dtype=jnp.int32),
            network_infos=network_infos,
            window_ptr=0,
        )

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        state: DNV2EstimatorState,
        obs: Observation,
    ):
        """Update DN v2 estimator using network interference structure."""
        # Extract observation info
        cluster_id = obs.design_info.cluster_id
        p = obs.design_info.p
        treatment = obs.action.astype(jnp.float32)
        reward = obs.reward

        # Update network info at current position
        new_network_info = self.network.get_network_info(env, env_params, obs)

        # DN-specific update logic
        z = treatment
        xi = z * (1 - p) / p + (1 - z) * p / (1 - p)
        update_val = xi * reward

        is_adjacent = jax.vmap(
            self.network.is_adjacent, in_axes=(None, None, 0, None)
        )(env, env_params, state.network_infos, new_network_info)

        # Update estimates for adjacent design clusters. The .at, .set logic
        # ensures that the current estimate counts only once for a given cluster
        estimates = state.estimates + (
            jnp.zeros((self.n_design_clusters,), dtype=jnp.float32)
            .at[state.design_cluster_ids]
            .set(jnp.where(is_adjacent, update_val, 0.0))
        )

        # These should be constant over time
        design_cluster_treatments = state.design_cluster_treatments.at[
            cluster_id
        ].set(treatment)
        design_cluster_treatment_probs = state.design_cluster_treatments.at[
            cluster_id
        ].set(p)
        design_cluster_counts = state.design_cluster_counts.at[cluster_id].add(
            1
        )

        # Replace info stored in the window with info from the current step
        design_cluster_ids = state.design_cluster_ids.at[current_ptr].set(
            cluster_id
        )
        network_infos = jax.tree.map(
            lambda a, b: a.at[current_ptr].set(b),
            state.network_infos,
            new_network_info,
        )

        return DNV2EstimatorState(
            design_cluster_treatments=design_cluster_treatments,
            design_cluster_treatment_probs=design_cluster_treatment_probs,
            design_cluster_counts=design_cluster_counts,
            estimates=estimates,
            design_cluster_ids=design_cluster_ids,
            network_infos=network_infos,
            window_ptr=(state.window_ptr + 1) % self.window_size,
        )

    def estimate(self, state: DNV2EstimatorState):
        """Compute DN treatment effect estimate."""
        N = state.design_cluster_counts.sum()
        mask = state.design_cluster_counts > 0
        z = state.design_cluster_treatments
        p = state.design_cluster_treatment_probs

        def compute_estimate():
            eta = z / p - (1 - z) / (1 - p)
            avg_y = (
                mask * state.estimates
            ).sum() / state.design_cluster_counts.sum()
            baseline = state.design_cluster_counts * avg_y
            return (mask * eta * (state.estimates - baseline)).sum() / N

        return jax.lax.cond(N > 0, lambda: compute_estimate(), lambda: 0.0)
