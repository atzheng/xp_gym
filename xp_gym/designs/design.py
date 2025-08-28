import operator as op
from typing import Tuple

from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import EnvParams, EnvState
import jax
import jax.numpy as jnp
from numpy import searchsorted

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
    time_attr: str = "t"
    frequency: float = 1.0

    def get_cluster_id(self, design_state: ClusterDesignState, env_state: EnvState) -> int:
        """
        Returns the cluster ID based on the time attribute.
        This design uses the time step as the cluster ID.
        """
        t = op.attrgetter(self.time_attr)(env_state)
        return jnp.floor(t / self.frequency).astype(jnp.int32)
