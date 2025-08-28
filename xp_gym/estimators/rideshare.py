from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import EnvParams
from jaxtyping import Float
from jax import Array

from or_gymnax.rideshare import RideshareEvent
from xp_gym.estimators.network import (
    SpatioTemporalInterferenceNetworkState,
)

@struct.dataclass
class RideshareInterferenceNetworkState:
    distance_matrix: Float[Array, "n_locations n_locations"]

@struct.dataclass
class RideshareInterferenceNetwork:
    time_threshold: int  # Time threshold for interference
    space_threshold: int

    def reset(
        self, rng: PRNGKey, env_params: EnvParams
    ) -> RideshareInterferenceNetworkState:
        return RideshareInterferenceNetworkState(
            space_ids=-jnp.ones(env_params.n_events, dtype=jnp.int32),
            time_ids=-jnp.ones(env_params.n_events, dtype=jnp.int32),
            distance_matrix=env_params.distances
        )

    def in_edges(
        self, state: SpatioTemporalInterferenceNetworkState, obs: Observation
    ) -> Integer[Array, "T"]:
        """
        Returns the indices of the in-edges for the rideshare network.
        """
        # Logic to determine in-edges based on time and space thresholds
        event: RideshareEvent = obs.state.event
        return jnp.where(
            (state.time_ids > 0) &
            (state.time_ids + self.time_threshold <= event.t) &
            self.space_adj[state.space_ids, event.src] <= self.space_threshold
        )[0]


    def update(
        self, state: SpatioTemporalInterferenceNetworkState, obs: Observation
    ) -> InterferenceNetworkState:
        """
        Update the network state based on the new observation.
        """
        # Update the state based on the new observation
        event: RideshareEvent = obs.state.event
        time_idx = obs.state.time
        return SpatioTemporalInterferenceNetworkState(
            space_ids=state.space_ids.at[time_idx].set(event.src),
            time_ids=state.time_ids.at[time_idx].set(event.t),
            distance_matrix=state.distance_matrix
        )
