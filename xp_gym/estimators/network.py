from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import Environment, EnvParams
from jaxtyping import Float, Integer, Bool
from typing import Tuple
from jax import Array
import jax.numpy as jnp
import jax
from dataclasses import field

from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.observation import Observation
from or_gymnax.rideshare import obs_to_state
from jaxtyping import Bool, Float, Integer


# General inteference networks
# -------------------------------------------------------------------------
@struct.dataclass
class NetworkInfo:
    """
    Abstract class to store information from an observation, as it
    relates to the interference network (e.g., the ID of the cluster it
    belongs to)
    """

    pass


@struct.dataclass
class InterferenceNetworkState:
    """
    Arbitrary information to store the state of the interference network.
    """

    pass


@struct.dataclass
class InterferenceNetwork:
    """
    A collection of methods that define the structure of an interference network.

    At minimum, this defines:

    1) get_network_info: extracts information from an observation
         required to determine which other observation it interferes with
         (e.g., an identifier of its corresponding node in the
          interference network)

    2) is_adjacent: given two NetworkInfo objects, determine whether
         they represent adjacent nodes
    """

    def reset(
        self, rng: PRNGKey, env, env_params: EnvParams
    ) -> InterferenceNetworkState:
        return InterferenceNetworkState()

    def is_adjacent(self, env, env_params, network_state: InterferenceNetworkState, x: NetworkInfo, y: NetworkInfo):
        """
        Determine if two network info objects are adjacent in the interference network.
        """
        raise NotImplementedError()

    def get_network_info(self, env, env_params, network_state: InterferenceNetworkState, obs: Array) -> NetworkInfo:
        """
        Extract network-related information from an observation.
        By default, simply returns the observation.
        """
        return obs

    def update(
        self, state: InterferenceNetworkState, obs: Observation
    ) -> InterferenceNetworkState:
        """
        Update the network state based on the new observation.
        """
        # Update the state based on the new observation
        return state


@struct.dataclass
class EmptyInterferenceNetwork:
    """A network object representing a graph with no edges; for sanity checks"""

    def reset(self, rng: PRNGKey, env, env_params: EnvParams) -> InterferenceNetworkState:
        return InterferenceNetworkState()

    def is_adjacent(self, env, env_params, network_state: InterferenceNetworkState, x: NetworkInfo, y: NetworkInfo):
        return False

    def get_network_info(self, env, env_params, network_state: InterferenceNetworkState, obs: Array) -> NetworkInfo:
        # Just a dummy object
        return jnp.zeros(1)

    def update(self, state: InterferenceNetworkState, obs: Observation) -> InterferenceNetworkState:
        return state


@struct.dataclass
class CompleteInterferenceNetwork:
    """A network object representing a complete graph"""

    def reset(self, rng: PRNGKey, env, env_params: EnvParams) -> InterferenceNetworkState:
        return InterferenceNetworkState()

    def is_adjacent(self, env, env_params, network_state: InterferenceNetworkState, x: NetworkInfo, y: NetworkInfo):
        return True

    def get_network_info(self, env, env_params, network_state: InterferenceNetworkState, obs: Array) -> NetworkInfo:
        # Just a dummy object
        return jnp.zeros(1)

    def update(self, state: InterferenceNetworkState, obs: Observation) -> InterferenceNetworkState:
        return state


# Rideshare-specific inteference networks
# -------------------------------------------------------------------------
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
    
    def reset(self, rng: PRNGKey, env, env_params: EnvParams) -> InterferenceNetworkState:
        return InterferenceNetworkState()

    def update(self, state: InterferenceNetworkState, obs: Observation) -> InterferenceNetworkState:
        return state

    def get_network_info(
        self, env: Environment, env_params: EnvParams, network_state: InterferenceNetworkState, obs: Array
    ) -> RideshareNetworkInfo:
        """Extract cluster info (lat, lng, t) from observation."""
        event, _, _ = obs_to_state(env_params.env_params.n_cars, obs)
        return RideshareNetworkInfo(time=event.t, location=event.src)

    def is_adjacent(
        self,
        env: Environment,
        env_params: EnvParams,
        network_state: InterferenceNetworkState,
        x: RideshareNetworkInfo,
        y: RideshareNetworkInfo,
    ):
        """Check if the edge (x, y) exists in the interference graph (i.e., x affects y)."""
        is_space_adj = (
            (
                env_params.env_params.distances[x.location, y.location]
                * 9
                / 1000
            )  # Matrix is in units of seconds, assuming 9 m/s driving
            <= self.max_spatial_distance
        )
        is_time_adj = ((y.time - x.time) < self.lookahead_steps) & (
            y.time > x.time
        )
        return is_time_adj & is_space_adj


# Estimator Base Class
# -------------------------------------------------------------------------
@struct.dataclass
class LimitedMemoryNetworkEstimatorState(EstimatorState):
    t: int
    design_cluster_treatments: Bool[Array, "n_design_cluster_ids"]
    design_cluster_treatment_probs: Float[Array, "n_design_cluster_ids"]
    design_cluster_ids: Integer[Array, "window_size"]
    network_infos: NetworkInfo
    network_state: InterferenceNetworkState
    estimate: Float[Array, ""]


@struct.dataclass
class LimitedMemoryNetworkEstimator(Estimator):
    """
    Base class for an estimator that computes interference based on a
    limited memory window.

    Many exact estimators for interference need to store all
    observations in order to compute interference effects;
    this approximates that behavior by assuming there is no interference
    outside of a fixed window size.

    This approximation be removed by setting the window size sufficiently large.
    """

    network: InterferenceNetwork
    window_size: int

    def reset(self, rng, env, env_params):
        dummy_obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
        network_state = self.network.reset(rng, env, env_params)
        network_infos = jax.vmap(
            lambda _: self.network.get_network_info(env, env_params, network_state, dummy_obs)
        )(
            jnp.zeros((self.window_size,))  # Dummy initialization
        )

        return LimitedMemoryNetworkEstimatorState(
            t=0,
            design_cluster_treatments=jnp.zeros(
                (self.window_size,), dtype=jnp.bool_
            ),
            design_cluster_treatment_probs=jnp.ones(
                (self.window_size,), dtype=jnp.bool_
            )
            * 0.5,
            design_cluster_ids=jnp.zeros((self.window_size,), dtype=jnp.int32),
            network_infos=network_infos,
            network_state=network_state,
            estimate=jnp.array(0.0, dtype=jnp.float32),
        )

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        state: LimitedMemoryNetworkEstimatorState,
        obs: Observation,
    ):
        # Extract observation info
        cluster_id = obs.design_info.cluster_id
        p = obs.design_info.p
        treatment = obs.action.astype(jnp.float32)
        reward = obs.reward
        z = treatment
        new_network_info = self.network.get_network_info(
            env, env_params, state.network_state, obs.obs
        )

        # Replace info stored in the window with info from the current step
        # Update network info at current position
        # -------------------------------------------------------------------------
        window_ptr = state.t % self.window_size
        design_cluster_ids = state.design_cluster_ids.at[window_ptr].set(
            cluster_id
        )
        zc = state.design_cluster_treatments
        pc = state.design_cluster_treatment_probs
        design_cluster_treatments = zc.at[window_ptr].set(z)
        design_cluster_treatment_probs = pc.at[window_ptr].set(p)
        network_infos = jax.tree.map(
            lambda a, b: a.at[window_ptr].set(b),
            state.network_infos,
            new_network_info,
        )
        
        # Update network state
        updated_network_state = self.network.update(state.network_state, obs)

        return LimitedMemoryNetworkEstimatorState(
            t=state.t + 1,
            design_cluster_treatments=design_cluster_treatments,
            design_cluster_treatment_probs=design_cluster_treatment_probs,
            design_cluster_ids=design_cluster_ids,
            network_infos=network_infos,
            network_state=updated_network_state,
            estimate=state.estimate,
        )

    def interference_mask(
        self,
        env: Environment,
        env_params: EnvParams,
        state: LimitedMemoryNetworkEstimatorState,
        obs: Observation,
    ) -> Bool[Array, "self.window_size"]:
        """
        Returns a mask of length self.window_size, where each
        unique cluster in the window which may interfere with obs
        will have exactly one "True" entry in the mask,
        corresponding to an arbitrary observation
        in the window from that cluster.

        This is commonly needed for interference estimators.
        """
        cluster_id = obs.design_info.cluster_id
        new_network_info = self.network.get_network_info(
            env, env_params, state.network_state, obs.obs
        )

        # Determine which other elements of the window represent
        # interfering observations
        is_adjacent = jax.vmap(
            self.network.is_adjacent, in_axes=(None, None, None, 0, None)
        )(env, env_params, state.network_state, state.network_infos, new_network_info)
        # Ensures that dummy entries (during first window) are ignored
        not_dummy = jnp.arange(self.window_size) <= state.t
        is_different_cluster = state.design_cluster_ids != cluster_id
        can_interfere = is_adjacent & not_dummy & is_different_cluster
        vals, ix = jnp.unique(
            jnp.where(can_interfere, state.design_cluster_ids, -1),
            return_index=True,
            size=self.window_size,
            fill_value=-1,
        )
        mask = jnp.zeros(self.window_size + 1).at[ix].set(vals != -1)[:-1]
        return mask
