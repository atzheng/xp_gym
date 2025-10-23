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
    time: Float[Array, ""] = field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float32)
    )
    location: Integer[Array, ""] = field(
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
        self, env: Environment, env_params: EnvParams, obs: Array
    ) -> RideshareNetworkInfo:
        """Extract cluster info (lat, lng, t) from observation."""
        event, _, _ = obs_to_state(env_params.env_params.n_cars, obs)
        return RideshareNetworkInfo(time=event.t, location=event.src)

    def is_adjacent(
        self,
        env: Environment,
        env_params: EnvParams,
        x: RideshareNetworkInfo,
        y: RideshareNetworkInfo,
    ):
        """Check if the edge (x, y) exists in the interference graph (i.e., x affects y)."""
        is_space_adj = (
            env_params.env_params.distances[x.location, y.location]
            <= self.max_spatial_distance
        )
        is_time_adj = ((y.time - x.time) < self.lookahead_steps) & (
            y.time > x.time
        )
        return is_time_adj & is_space_adj


@struct.dataclass
class DNV2EstimatorState(EstimatorState):
    """State for DN v2 estimator using network abstractions."""

    t: int
    design_cluster_treatments: Bool[Array, "n_design_cluster_ids"]
    design_cluster_treatment_probs: Float[Array, "n_design_cluster_ids"]
    design_cluster_ids: Integer[Array, "window_size"]
    network_infos: NetworkInfo
    estimate: Float[Array, ""]


@struct.dataclass
class DNV2Estimator(Estimator):
    """
    The DN-Cluster estimator for clustering.

    Note the *Limited memory window*:
    The exact DN estimator would need to store all past
    observations to compute interference effects.
    Here, we limit to a fixed window size.
    This approximation can be removed by setting the window size
    to be the horizon.
    """

    network: InterferenceNetwork
    window_size: int

    def reset(self, rng, env, env_params):
        """Initialize DN v2 estimator with network state."""
        dummy_obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
        network_infos = jax.vmap(
            lambda _: self.network.get_network_info(env, env_params, dummy_obs)
        )(
            jnp.zeros((self.window_size,))  # Dummy initialization
        )

        return DNV2EstimatorState(
            t=0,
            design_cluster_treatments=jnp.zeros(
                (self.window_size,), dtype=jnp.bool_
            ),
            design_cluster_treatment_probs=jnp.zeros(
                (self.window_size,), dtype=jnp.bool_
            ),
            design_cluster_ids=jnp.zeros((self.window_size,), dtype=jnp.int32),
            network_infos=network_infos,
            estimate=jnp.array(0.0, dtype=jnp.float32),
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
        new_network_info = self.network.get_network_info(
            env, env_params, obs.obs
        )

        # Determine which other elements of the window represent
        # interfering observations
        # -------------------------------------------------------------------------
        is_adjacent = jax.vmap(
            self.network.is_adjacent, in_axes=(None, None, 0, None)
        )(env, env_params, state.network_infos, new_network_info)
        # Ensures that dummy entries (during first window) are ignored
        not_dummy = jnp.arange(self.window_size) <= state.t
        # Each cluster should be represented by its first occurrence
        is_first = is_first_occurrence(state.design_cluster_ids)

        zc = state.design_cluster_treatments
        pc = state.design_cluster_treatment_probs

        # Compute update (Eqn. (9) from paper, DN-Cluster estimator)
        # -------------------------------------------------------------------------
        z = treatment
        xi = z * (1 - p) / p + (1 - z) * p / (1 - p)
        #
        estimate = (
            state.estimate
            + jnp.sum(
                (is_adjacent & not_dummy & is_first)
                * (zc / pc - (1 - zc) / (1 - pc))
                * xi
                + z / p
                - (1 - z) / (1 - p)
            )
            * reward
        )

        # Replace info stored in the window with info from the current step
        # Update network info at current position
        # -------------------------------------------------------------------------
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

    def estimate(self, env, env_params, state: DNV2EstimatorState):
        """Compute DN treatment effect estimate."""
        return state.estimate / state.t
