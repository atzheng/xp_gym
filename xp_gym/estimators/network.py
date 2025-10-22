from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import EnvParams
from jaxtyping import Float, Integer, Bool
from jax import Array
import jax.numpy as jnp

from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.designs.design import Design
from xp_gym.observation import Observation


@struct.dataclass
class InterferenceNetworkState:
    pass


@struct.dataclass
class InterferenceNetwork:
    """
    A collection of methods that define the structure of an interference network.
    """

    def reset(
        self, rng: PRNGKey, env, env_params: EnvParams
    ) -> InterferenceNetworkState:
        return InterferenceNetworkState()
        # Initialize any graph structure here

    def in_edges(
        self, state: InterferenceNetworkState, obs: Observation
    ) -> Bool[Array, "n_edges"]:
        raise NotImplementedError()

    def out_edges(
        self, state: InterferenceNetworkState, obs: Observation
    ) -> Bool[Array, "n_edges"]:
        raise NotImplementedError()

    def update(
        self, state: InterferenceNetworkState, obs: Observation
    ) -> InterferenceNetworkState:
        """
        Update the network state based on the new observation.
        """
        # Update the state based on the new observation
        return state


@struct.dataclass
class SpatioTemporalInterferenceNetworkState(InterferenceNetworkState):
    space_ids: Integer[Array, "T"]
    time_ids: Integer[Array, "T"]


@struct.dataclass
class SpatioTemporalInterferenceNetwork(object):
    T: int  # Time horizon

    def reset(self, rng: PRNGKey, env, env_params: EnvParams):
        return SpatioTemporalInterferenceNetworkState(
            space_ids=-jnp.ones(self.T, dtype=jnp.int32),
            time_ids=-jnp.ones(self.T, dtype=jnp.int32),
        )

    def in_edges(
        self, state: SpatioTemporalInterferenceNetworkState, obs: Observation
    ) -> Integer[Array, "T"]:
        raise NotImplementedError()

    def get_space_id(self, obs: Observation) -> int:
        raise NotImplementedError()

    def get_time_id(self, obs: Observation) -> int:
        raise NotImplementedError()

    def update(
        self, state: SpatioTemporalInterferenceNetworkState, obs: Observation
    ) -> SpatioTemporalInterferenceNetworkState:
        """
        Update the network state based on the new observation.
        """
        # Update the state based on the new observation
        return SpatioTemporalInterferenceNetworkState(
            space_ids=state.space_ids.at[obs.t].set(self.get_space_id(obs)),
            time_ids=state.time_ids.at[obs.t].set(self.get_time_id(obs)),
        )


@struct.dataclass
class ClusterNetworkEstimatorState(Estimator):
    network_state: InterferenceNetworkState
    cluster_ids: Integer[Array, "T"]  # Cluster IDs for each time step
    treatments: Integer[Array, "T"]  # Treatments assigned at each time step
    ps: Float[
        Array, "T"
    ]  # Probabilities of treatment assignment at each time step


@struct.dataclass
class ClusterNetworkEstimator(Estimator):
    """
    Estimator that accounts for interference in a known network structure,
    """

    network: InterferenceNetwork
    T: int  # Time horizon
    # Need to somehow compute the probability of seeing arbitrary treatment vectors

    def reset(self, rng: PRNGKey, env, env_params: EnvParams):
        """
        Initialize the network estimator with necessary parameters.
        """
        # Initialize any state variables needed for the estimator
        return ClusterNetworkEstimatorState(
            network_state=self.network.reset(rng, env_params),
            cluster_ids=-jnp.ones(self.T, dtype=jnp.int32),
            treatments=-jnp.ones(self.T, dtype=jnp.int32),
            ps=-jnp.ones(self.T, dtype=jnp.float32),
        )

    def update(
        self, state: ClusterNetworkEstimatorState, obs: Observation
    ) -> ClusterNetworkEstimatorState:
        """
        Update the estimator with new data, considering network interference.
        """
        # Update the state based on the new observation
        cluster_id = obs.design_info.cluster_id
        p = obs.design_info.cluster_id
        new_network_state = self.network.update(state.network_state, obs)
        t = obs.state.time
        return ClusterNetworkEstimatorState(
            network_state=new_network_state,
            cluster_ids=state.cluster_ids.at[t].set(cluster_id),
            treatments=state.treatments.at[t].set(obs.action),
            ps=state.ps.at[t].set(p),
        )


@struct.dataclass
class HTClusterNetworkEstimatorState(ClusterNetworkEstimatorState):
    estimate: Float[Array, ""]  # Current estimate of the treatment effect


@struct.dataclass
class HTClusterNetworkEstimator(ClusterNetworkEstimator):
    """
    Horvitz-Thompson estimator that accounts for interference in a known network structure,
    """

    def reset(self, rng: PRNGKey, env, env_params: EnvParams):
        """
        Initialize the HT network estimator with necessary parameters.
        """
        base_state = super().reset(rng, env_params)
        return HTClusterNetworkEstimatorState(
            network_state=base_state.network_state,
            cluster_ids=base_state.cluster_ids,
            treatments=base_state.treatments,
            ps=base_state.ps,
            estimate=jnp.zeros(1)
        )

    def update(self, state: HTClusterNetworkEstimatorState, obs):
        new_base_state = super().update(state, obs)
        is_in_edge = self.network.in_edges(state.network_state, obs)
        is_tr = jnp.all(jnp.where(is_in_edge, state.treatments, False))
        p_tr = 0.  # FIXME
        is_co = jnp.all(jnp.where(is_in_edge, ~state.treatments, False))
        p_co = 0.
        return HTClusterNetworkEstimatorState(
            network_state=new_base_state.network_state,
            cluster_ids=new_base_state.cluster_ids,
            treatments=new_base_state.treatments,
            ps=new_base_state.ps,
            estimate=(is_tr / p_tr - is_co / p_co) * obs.reward + state.estimate,
        )

    def estimate(self, state: HTClusterNetworkEstimatorState):
        """
        Estimate the value based on the current state of the estimator.
        """
        return state.estimate
