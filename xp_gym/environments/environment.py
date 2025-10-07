from dataclasses import field
from flax import struct
from gymnax.environments import environment
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Tuple, Optional
import chex


@struct.dataclass
class XPEnvParams(environment.EnvParams):
    policy_A_kwargs: Dict = field(default_factory=dict)
    policy_B_kwargs: Dict = field(default_factory=dict)
    env_params: environment.EnvParams = environment.EnvParams()


class XPEnvironment(environment.Environment):
    """
    Minimal wrapper to run an experiment between two policies A and B in a
    gymnax environment. Essentially, reduces the action space to two actions:
    0 for policy A and 1 for policy B, and stores
    """

    def __init__(
        self,
        env: environment.Environment,
        policy_A: Callable,
        policy_B: Callable,
    ):
        super(XPEnvironment, self).__init__()
        self.env = env
        self.policy_A = policy_A
        self.policy_B = policy_B

    def step_env(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: int,
        params: XPEnvParams,
    ) -> Tuple[
        chex.Array,
        environment.EnvState,
        jnp.ndarray,
        jnp.ndarray,
        Dict[Any, Any],
    ]:
        key, policy_key = jax.random.split(key, 2)
        key, step_key = jax.random.split(key, 2)
        env_params = params.env_params
        obs = self.env.get_obs(state, env_params)
        action_B = self.policy_B.apply(
            env_params, dict(), obs, policy_key, **params.policy_B_kwargs
        )
        action_A = self.policy_A.apply(
            env_params, dict(), obs, policy_key, **params.policy_A_kwargs
        )
        action, action_info = jax.lax.cond(
            action, lambda: action_B, lambda: action_A
        )
        next_obs, next_state, reward, done, info = self.env.step(
            step_key, state, action, env_params
        )

        return (
            next_obs,
            next_state,
            reward,
            done,
            {"action_A": action_A, "action_B": action_B, **info},
        )

    @property
    def num_actions(self):
        return 2

    def action_space(self):
        return spaces.Discrete(2)

    # Pass through methods for observation and state spaces
    def observation_space(self):
        return self.env.observation_space()

    def state_space(self):
        return self.env.state_space()

    def reset_env(self, key: chex.PRNGKey, params: XPEnvParams):
        key, reset_key = jax.random.split(key, 2)
        obs, state = self.env.reset(reset_key, params.env_params)
        return obs, state

    @property
    def default_params(self):
        env_defaults = self.env.default_params
        return XPEnvParams(
            max_steps_in_episode=env_defaults.max_steps_in_episode,
            policy_A_kwargs={},
            policy_B_kwargs={},
            env_params=self.env.default_params,
        )
