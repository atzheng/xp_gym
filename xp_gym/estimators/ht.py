from flax import struct
import jax.numpy as jnp

from xp_gym.estimators.network import (
    LimitedMemoryNetworkEstimator,
    LimitedMemoryNetworkEstimatorState,
)
from xp_gym.observation import Observation
from gymnax.environments.environment import Environment, EnvParams
from jax import Array


@struct.dataclass
class LimitedMemoryHTEstimator(LimitedMemoryNetworkEstimator):
    """
    The Horvitz-Thompson estimator for experiments with clustered designs
    and a known inteference graph.

    WARNING: This implementation implicitly assumes that an experimental
    unit A can only suffer interference effects from another unit B if
    B arrives in some fixed window BEFORE A.
    """

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        state: LimitedMemoryNetworkEstimatorState,
        obs: Observation,
    ):
        """Update DN v2 estimator using network interference structure."""
        # Extract observation info
        cluster_id = obs.design_info.cluster_id
        p = obs.design_info.p
        z = obs.action.astype(jnp.float32)
        reward = obs.reward

        zc = state.design_cluster_treatments
        pc = state.design_cluster_treatment_probs
        interference_mask = self.interference_mask(env, env_params, state, obs)

        all_tr_ipw = (
            jnp.prod(jnp.where(interference_mask, zc / pc, 1.0)) * z / p
        )
        all_co_ipw = (
            jnp.prod(jnp.where(interference_mask, (1 - zc) / (1 - pc), 1.0))
            * (1 - z)
            / (1 - p)
        )
        ipw = all_tr_ipw - all_co_ipw
        estimate = state.estimate + ipw * reward
        return (
            super()
            .update(env, env_params, state, obs)
            .replace(estimate=estimate)
        )

    def estimate(
        self, env, env_params, state: LimitedMemoryNetworkEstimatorState
    ):
        return state.estimate / state.t
