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
    NetworkInfo,
)
from xp_gym.observation import Observation
from gymnax.environments.environment import Environment, EnvParams
from or_gymnax.rideshare import ManhattanRidesharePricing, obs_to_state
from jaxtyping import Array, Bool, Float, Integer
from jax import Array


def is_first_occurrence(x):
    """
    Returns a boolean array indicating whether each element in x
    is the first occurrence of that element.
    """
    asort = jnp.argsort(x)
    xsort = x[asort]
    return ((xsort - jnp.roll(xsort, 1)) != 0)[jnp.argsort(x)]


@struct.dataclass
class RideshareNetworkInfo:
    time: Float[Array, ()] = field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float32)
    )
    location: Integer[Array, ()] = field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int32)
    )


@struct.dataclass
class RideshareNetwork(InterferenceNetwork):
    """
    Concrete implementation of spatiotemporal interference network, where
    adjacency is defined based on spatial distance (km) and temporal distance (steps).
    """

    lookahead_steps: int = 600
    max_spatial_distance: int = 2  # km

    def get_network_info(
        self, env: Environment, env_params: EnvParams, obs: Observation
    ) -> RideshareNetworkInfo:
        """Extract cluster info (lat, lng, t) from observation."""
        state = obs_to_state(env, env_params, obs)
        return RideshareNetworkInfo(
            time=state.event.t, location=state.event.src
        )

    def is_adjacent(
        self,
        env: Environment,
        env_params: EnvParams,
        x: RideshareNetworkInfo,
        y: RideshareNetworkInfo,
    ):
        """Check if the edge (x, y) exists in the interference graph (i.e., x affects y)."""
        is_space_adj = (
            env_params.distance[x.location, y.location]
            <= self.max_spatial_distance
        )
        is_time_adj = y.time - x.time < self.lookahead_steps & y.time > x.time
        return is_time_adj & is_space_adj


@struct.dataclass
class DNV2EstimatorState(EstimatorState):
    """State for DN v2 estimator using network abstractions."""

    t: int
    design_cluster_treatments: Bool[Array, "n_design_cluster_ids"]
    design_cluster_treatment_probs: Float[Array, "n_design_cluster_ids"]
    design_cluster_ids: Integer[Array, "window_size"]
    network_infos: NetworkInfo
    estimate: Float[Array, ()]


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
        network_infos = jax.vmap(
            lambda _: self.network.get_network_info(
                env, env_params, env.observation_space.sample()
            )
        )(
            jnp.zeros((self.window_size,))  # Dummy initialization
        )

        return DNV2EstimatorState(
            t=0,
            design_cluster_treatments=jnp.zeros((self.window_size,), dtype=jnp.bool_),
            design_cluster_treatment_probs=jnp.zeros((self.window_size,), dtype=jnp.bool_),
            design_cluster_ids=jnp.zeros((self.window_size,), dtype=jnp.int32),
            network_infos=network_infos,
            estimate=jnp.array(0.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: DNV2EstimatorState,
        obs: Observation,
        env: Environment,
        env_params: EnvParams,
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
        # Ensures that dummy entries (during first window) are ignored
        not_dummy = jnp.arange(self.window_size) <= state.t  
        # Each cluster should be represented by its first occurrence
        is_first = is_first_occurrence(state.design_cluster_ids)  

        zc = state.design_cluster_treatments
        pc = state.design_cluster_treatment_probs

        # Eqn. (9) from paper (DN-Cluster estimator)
        estimate = state.estimate + (
            (is_adjacent & not_dummy & is_first_occurrence & is_first) * (
                zc / pc - (1 - zc) / (1 - pc)
            ) * xi
            + z / p - (1 - z) / (1 - p)
       ) * reward

        # Replace info stored in the window with info from the current step
        window_ptr = state.t % self.window_size
        design_cluster_ids = state.design_cluster_ids.at[window_ptr].set(
            cluster_id
        )
        design_cluster_treatments = zc.at[window_ptr].set(z)
        design_cluster_treatment_probs = pc.at[window_ptr].set(p)
        network_infos = jax.tree.map(
            lambda a, b: a.at[window_ptr].set(b),
            state.network_infos,
            new_network_info,
        )

        return DNV2EstimatorState(
            t=state.t + 1,
            design_cluster_treatments=design_cluster_treatments,
            design_cluster_treatment_probs=design_cluster_treatment_probs,
            design_cluster_ids=design_cluster_ids,
            network_infos=network_infos,
            estimate=estimate,
        )

    def estimate(self, state: DNV2EstimatorState, env, env_params):
        """Compute DN treatment effect estimate."""
        return state.estimate / state.t

