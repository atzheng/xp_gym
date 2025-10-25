from flax import struct
import jax.numpy as jnp

from xp_gym.estimators.network import (
    LimitedMemoryNetworkEstimator,
    LimitedMemoryNetworkEstimatorState,
)
from xp_gym.observation import Observation
from xp_gym.estimators.estimator import Estimator, EstimatorState
from gymnax.environments.environment import Environment, EnvParams
from jax import Array
from jaxtyping import Float



@struct.dataclass
class DNEstimatorState(EstimatorState):
    limited_memory_state: LimitedMemoryNetworkEstimatorState
    sum_rewards: Float[Array, ""]  # For double robust baseline

    
@struct.dataclass
class LimitedMemoryDNEstimator(LimitedMemoryNetworkEstimator):
    """
    The DN-Cluster estimator for interference in experiments with clustered
    designs and a known interference graph.

    WARNING: This implementation implicitly assumes that an experimental
    unit A can only suffer interference effects from another unit B if
    B arrives in some fixed window BEFORE A.
    """
    def reset(self, rng, env, env_params):
        """Initialize DN v2 estimator with network state."""
        lms = super().reset(rng, env, env_params)
        return DNEstimatorState(limited_memory_state=lms, sum_rewards=0.)

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        state: DNEstimatorState,
        obs: Observation,
    ):
        # Extract observation info
        p = obs.design_info.p
        treatment = obs.action.astype(jnp.float32)
        reward = obs.reward
        lms = state.limited_memory_state
        zc = lms.design_cluster_treatments
        pc = lms.design_cluster_treatment_probs
        interference_mask = self.interference_mask(env, env_params, lms, obs)
        baseline = state.sum_rewards / (lms.t + 1)    # use avg reward as doubly robust baseline

        z = treatment
        xi = z * (1 - p) / p + (1 - z) * p / (1 - p)

        estimate = (
            lms.estimate
            + jnp.sum(
                interference_mask * (
                    (zc / pc - (1 - zc) / (1 - pc)) * (xi * reward - baseline)
                )
            )
            + (z / p - (1 - z) / (1 - p)) * (reward - baseline)
        )

        new_lms = (
            super()
            .update(env, env_params, lms, obs)
            .replace(estimate=estimate)
        )
        return DNEstimatorState(
            limited_memory_state=new_lms,
            sum_rewards=state.sum_rewards + reward,
        )

    def estimate(
        self, env, env_params, state: DNEstimatorState
    ):
        ss = state.limited_memory_state
        return ss.estimate / ss.t
