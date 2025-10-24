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
from jaxtyping import Integer
from jax import Array

from xp_gym.observation import Observation
from importlib import resources as impresources
from .. import data


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

    def assign_treatment(
        self, design_state: ClusterDesignState, env_state: EnvState
    ):
        cluster_id = self.get_cluster_id(design_state, env_state)
        new_rng = jax.random.fold_in(design_state.rng, cluster_id)
        z = jax.random.bernoulli(new_rng, self.p)
        return (
            z,
            ClusterDesignInfo(p=self.p, cluster_id=cluster_id),
        )


@struct.dataclass
class UnitRandomizedDesignState(ClusterDesignState):
    t: int


@struct.dataclass
class UnitRandomizedDesign(ClusterRandomizedDesign):
    def reset(self, rng: PRNGKey, env_params: EnvParams) -> ClusterDesignState:
        return UnitRandomizedDesignState(t=0, rng=rng)

    def get_cluster_id(self, _, env_state) -> bool:
        # Unique cluster at every time step
        return env_state.time

    def update(
        self, state: UnitRandomizedDesignState, obs: Observation
    ) -> UnitRandomizedDesignState:
        return UnitRandomizedDesignState(t=state.t + 1, rng=state.rng)


@struct.dataclass
class SwitchbackDesign(ClusterRandomizedDesign):
    time_attr: str = "event.t"
    switch_every: float = 1.0

    def get_cluster_id(
        self, design_state: ClusterDesignState, env_state: EnvState
    ) -> int:
        """
        Returns the cluster ID based on the time attribute.
        This design uses the time step as the cluster ID.
        """
        t = op.attrgetter(self.time_attr)(env_state)
        return (t // self.switch_every).astype(jnp.int32)


@struct.dataclass
class SpatioTemporalClusterDesignInfo(ClusterDesignInfo):
    """
     Extended design info that includes spatial mapping information for DN
    ."""

    time_cluster: int  # Time-based cluster component
    space_cluster: int  # Space-based cluster component


def load_rideshare_clusters():
    """
    Load rideshare spatial clusters.
    """
    zones_file = impresources.files(data) / "taxi-zones.parquet"
    nodes_file = impresources.files(data) / "manhattan-nodes.parquet"
    zones = pd.read_parquet(zones_file)
    unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
    zones["zone_id"] = unq_zone_ids
    nodes = pd.read_parquet(nodes_file)
    nodes["lng"] = nodes["lng"].astype(float)
    nodes["lat"] = nodes["lat"].astype(float)
    nodes_zones = nodes.merge(zones, on="osmid")

    # Convert to jax array
    max_src_idx = nodes_zones.index.max()
    src_to_zone = np.full(max_src_idx + 1, -1, dtype=np.int32)
    for idx, row in nodes_zones.iterrows():
        src_to_zone[idx] = row["zone_id"]

    return jnp.array(src_to_zone)


@struct.dataclass
class RideshareClusterDesignState(ClusterDesignState):
    """
    State for the spatiotemporal cluster design.
    """

    src_to_zone: Integer[Array, "n_nodes"]
    n_zones: Integer[Array, ""]


@struct.dataclass
class RideshareClusterDesign(ClusterRandomizedDesign):
    """
    Spatiotemporal cluster design, with randomization based on
    NYC TLC taxi zones x Time.
    """

    switch_every: int = 5000

    def reset(
        self, rng: PRNGKey, env_params: EnvParams
    ) -> RideshareClusterDesignState:
        src_to_zone = load_rideshare_clusters()
        return RideshareClusterDesignState(
            rng=rng,
            src_to_zone=src_to_zone,
            n_zones=src_to_zone.max() + 1,
        )

    def get_cluster_id(
        self,
        design_state: RideshareClusterDesignState,
        env_state: EnvState,
    ) -> int:
        time_id = (
            env_state.event.t // self.switch_every + 1
        ) * self.switch_every  # Identifies the end of the period
        space_id = design_state.src_to_zone[env_state.event.src]
        cluster_id = time_id * design_state.n_zones + space_id
        return cluster_id

    def assign_treatment(
        self,
        design_state: RideshareClusterDesignState,
        env_state: EnvState,
    ):
        """Assign treatment with efficient O(1) cluster ID calculation."""
        z, base_design_info = super().assign_treatment(design_state, env_state)

        # Extract components for design info
        cluster_id = base_design_info.cluster_id
        n_zones = design_state.n_zones
        time_cluster = cluster_id // n_zones
        space_cluster = cluster_id % n_zones

        design_info = SpatioTemporalClusterDesignInfo(
            p=self.p,
            cluster_id=cluster_id,
            time_cluster=time_cluster,
            space_cluster=space_cluster,
        )

        return z, design_info
