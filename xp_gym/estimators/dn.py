from flax import struct
import jax.numpy as jnp

from xp_gym.estimators.network import (
    LimitedMemoryNetworkEstimator,
    LimitedMemoryNetworkEstimatorState,
)
from xp_gym.observation import Observation
from xp_gym.estimators.estimator import Estimator
from gymnax.environments.environment import Environment, EnvParams



@struct.dataclass
class LimitedMemoryDNEstimator(LimitedMemoryNetworkEstimator):
    """
    The DN-Cluster estimator for interference in experiments with clustered
    designs and a known interference graph.

    WARNING: This implementation implicitly assumes that an experimental
    unit A can only suffer interference effects from another unit B if
    B arrives in some fixed window BEFORE A.
    """
    baseline: float = 0.0  # Baseline for simple doubly robust adjustment

    def reset(self, rng, env, env_params):
        """Initialize DN v2 estimator with network state."""
        return super().reset(rng, env, env_params)

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        state: LimitedMemoryNetworkEstimatorState,
        obs: Observation,
    ):
        # Extract observation info
        p = obs.design_info.p
        treatment = obs.action.astype(jnp.float32)
        reward = obs.reward
        zc = state.design_cluster_treatments
        pc = state.design_cluster_treatment_probs
        interference_mask = self.interference_mask(env, env_params, state, obs)

        z = treatment
        xi = z * (1 - p) / p + (1 - z) * p / (1 - p)

        estimate = (
            state.estimate
            + jnp.sum(
                interference_mask * (
                    (zc / pc - (1 - zc) / (1 - pc)) * (xi * reward - self.baseline)
                )
            )
            + (z / p - (1 - z) / (1 - p)) * (reward - self.baseline)
        )

        return (
            super()
            .update(env, env_params, state, obs)
            .replace(estimate=estimate)
        )

    def estimate(
        self, env, env_params, state: LimitedMemoryNetworkEstimatorState
    ):
        return state.estimate / state.t
