from flax import struct
import jax.numpy as jnp
import jax

from xp_gym.observation import Observation
from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.estimators.limited_memory import LimitedMemoryEstimatorState
from xp_gym.estimators.network import LimitedMemoryNetworkEstimator
from gymnax.environments.environment import Environment, EnvParams


@struct.dataclass
class LimitedMemoryDNEstimator(LimitedMemoryNetworkEstimator):
    """
    The DN-Cluster estimator for interference in experiments with clustered
    designs and a known interference graph.

    WARNING: This implementation implicitly assumes that an experimental
    unit A can only suffer interference effects from another unit B if
    B arrives in some fixed window BEFORE A.

    @param baseline: Baseline reward, used for a simple (constant) doubly robust
      adjustment. Reduces variance.
    @param use_known_actions: Whether to assume that a given Observation.info has
    fields "action_A" and "action_B" containing the true underlying actions
    (not just the treatments), as in environments.environment.XPEnvironment.
    This provides substantial variance reduction when treatment and control
    policies frequently choose the same underlying action. This can
    realistically be assumed to be observable in many real-world settings.
    """

    baseline: float = 0.0
    use_known_actions: bool = False

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

        if self.use_known_actions:
            # Identify which clusters in history would have taken different actions
            # under treatment vs. control policies
            is_different_action = (
                obs_mem.info["action_A"][0] != obs_mem.info["action_B"][0]
            )
        else:
            is_different_action = True

        # Extract observation info
        reward = obs.reward
        zc = obs_mem.action
        pc = obs_mem.design_info.p
        interference_mask = self.interference_mask(
            env, env_params, state_w_new_memory, obs, mask=is_different_action
        )

        ipwc = zc / pc - (1 - zc) / (1 - pc)
        new_estimate = state.estimator_state + jnp.sum(
            interference_mask * ipwc * (reward - self.baseline)
        )

        return state_w_new_memory.replace(estimator_state=new_estimate)

    def estimate(
        self, env, env_params, design, state: LimitedMemoryEstimatorState
    ):
        return state.estimator_state / state.t


