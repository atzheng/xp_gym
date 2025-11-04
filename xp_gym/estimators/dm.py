from flax import struct
import jax.numpy as jnp
import jax

from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.observation import Observation


@struct.dataclass
class DMEstimatorState(EstimatorState):
    treated_sum: float    # Sum of rewards for treated observations
    treated_count: int    # Count of treated observations  
    control_sum: float    # Sum of rewards for control observations
    control_count: int    # Count of control observations


@struct.dataclass
class DMEstimator(Estimator):
    """
    Simple difference in means estimator, without IPW.
    """
    def reset(self, rng, env, env_params, design):
        return DMEstimatorState(0.0, 0, 0.0, 0)

    def update(self, env, env_params, design, state: DMEstimatorState, obs: Observation):
        # Accumulate rewards by treatment group (like original naive_update)
        treated_sum = jnp.where(
            obs.action,
            state.treated_sum + obs.reward,
            state.treated_sum
        )
        treated_count = jnp.where(
            obs.action,
            state.treated_count + 1,
            state.treated_count
        )
        control_sum = jnp.where(
            obs.action,
            state.control_sum,
            state.control_sum + obs.reward
        )
        control_count = jnp.where(
            obs.action,
            state.control_count,
            state.control_count + 1
        )
        
        return DMEstimatorState(
            treated_sum, treated_count, control_sum, control_count
        )

    def estimate(self, env, env_params, design, state: DMEstimatorState):
        # Avoid division by zero
        treated_avg = jnp.where(
            state.treated_count > 0,
            state.treated_sum / state.treated_count,
            0.0
        )
        control_avg = jnp.where(
            state.control_count > 0,
            state.control_sum / state.control_count,
            0.0
        )
        
        # jax.debug.print("treated_avg = {}, control_avg = {}", treated_avg, control_avg)
        return treated_avg - control_avg
