from flax import struct
from gymnax.environments.environment import EnvState
from gymnax.environments import environment
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from typing import Any, Dict, Tuple
import chex


@struct.dataclass
class ABTestEnvParams(environment.EnvParams):
    arm_0_reward: float = 0.0
    arm_1_reward: float = 0.1
    noise_std: float = 1.0


class ABTestEnv(environment.Environment):
    """
    Simple A/B test environment with two arms.
    Action 0 corresponds to arm 0, action 1 corresponds to arm 1.
    Each arm gives a reward drawn from a normal distribution with configurable mean and noise.
    """

    def __init__(self):
        super(ABTestEnv, self).__init__()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: ABTestEnvParams,
    ) -> Tuple[
        chex.Array,
        EnvState,
        jnp.ndarray,
        jnp.ndarray,
        Dict[Any, Any],
    ]:
        # Get base reward based on action (arm selection)
        base_reward = jax.lax.cond(
            action == 1,
            lambda: params.arm_1_reward,
            lambda: params.arm_0_reward,
        )
        
        # Add noise to reward
        reward = base_reward + jax.random.normal(key) * params.noise_std
        
        # Update state
        new_state = EnvState(time=state.time + 1)

        # Observation is just the current time step (minimal)
        obs = jnp.array([state.time], dtype=jnp.float32)
        done = False
        return obs, new_state, reward, done, {}

    @property
    def num_actions(self):
        return 2

    def action_space(self, params: ABTestEnvParams):
        return spaces.Discrete(2)

    def observation_space(self, params: ABTestEnvParams):
        return spaces.Box(low=0, high=jnp.inf, shape=(1,), dtype=jnp.float32)

    def state_space(self, params: ABTestEnvParams):
        return spaces.Box(low=0, high=jnp.inf, shape=(), dtype=jnp.int32)

    def reset_env(self, key: chex.PRNGKey, params: ABTestEnvParams):
        state = EnvState(time=0)
        obs = jnp.array([0.0], dtype=jnp.float32)
        return obs, state

    @property
    def default_params(self):
        return ABTestEnvParams()
