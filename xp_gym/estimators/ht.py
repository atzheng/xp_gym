from flax import struct
import jax.numpy as jnp

from xp_gym.estimators.network import (
    LimitedMemoryNetworkEstimator,
)
from xp_gym.estimators.limited_memory import LimitedMemoryEstimatorState
from xp_gym.observation import Observation
from gymnax.environments.environment import Environment, EnvParams
from jax import Array


@struct.dataclass
class LimitedMemoryHTEstimator(LimitedMemoryNetworkEstimator):
    """
    The Horvitz-Thompson estimator for experiments with clustered designs
    and a known interference graph. Unbiased if the graph is correct.

    WARNING: This implementation implicitly assumes that an experimental
    unit A can only suffer interference effects from another unit B if
    B arrives in some fixed window BEFORE A.
    """

    baseline: float = 0.0  # Baseline for simple doubly robust adjustment

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        design,
        state: LimitedMemoryEstimatorState,
        obs: Observation,
    ):
        state_w_new_memory = super().update(env, env_params, design, state, obs)
        obs_mem = state_w_new_memory.obs

        # Extract observation info
        reward = obs.reward
        zc = obs_mem.action
        pc = obs_mem.design_info.p
        interference_mask = self.interference_mask(env, env_params, state, obs)

        ipw = jnp.prod(zc / pc, where=interference_mask) - jnp.prod(
            (1 - zc) / (1 - pc), where=interference_mask
        )
        estimate = state.estimator_state + ipw * (reward - self.baseline)
        return state_w_new_memory.replace(estimator_state=estimate)

    def estimate(
        self, env, env_params, design, state: LimitedMemoryEstimatorState
    ):
        return state.estimator_state / state.t
