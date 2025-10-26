#!/usr/bin/env python3
from flax import struct
from dataclasses import field
from gymnax.environments.environment import EnvState
from gymnax.environments import environment
from gymnax.environments import spaces
import jax
from jax import Array
from jaxtyping import Integer, Float
import jax.numpy as jnp
from typing import Any, Dict, Tuple
import chex


@struct.dataclass
class LimitedMemoryEnvState(environment.EnvState):
    z_history: Integer[Array, "window_size"]


@struct.dataclass
class LimitedMemoryEnvParams(environment.EnvParams):
    reward_tensor: Float[Array, "2 ..."] = field(
        default_factory=lambda: jnp.zeros((2, 2))
    )
    noise_std: float = 1.0

    @property
    def window_size(self):
        return len(self.reward_tensor.shape)


class LimitedMemoryEnv(environment.Environment):
    """
    Simple interference environment where the mean outcome at t
    depends arbitrarily on z_{t - window_size} ... z_t taken.
    """

    def step_env(
        self,
        key: chex.PRNGKey,
        state: LimitedMemoryEnvState,
        action: int,
        params: LimitedMemoryEnvParams,
    ) -> Tuple[
        chex.Array,
        LimitedMemoryEnvState,
        jnp.ndarray,
        jnp.ndarray,
        Dict[Any, Any],
    ]:
        # Update z_history with current action using roll
        new_z_history = jnp.roll(state.z_history, -1)
        new_z_history = new_z_history.at[-1].set(action)

        # Use flat indexing pattern to get reward from tensor
        # flat_idx = jnp.ravel_multi_index(
        #     tuple(new_z_history), params.reward_tensor.shape
        # )
        base_reward = params.reward_tensor[tuple(new_z_history)]

        # Add noise to reward
        reward = base_reward + jax.random.normal(key) * params.noise_std

        # Update state
        new_state = LimitedMemoryEnvState(
            time=state.time + 1, z_history=new_z_history
        )

        # Observation is the history
        obs = new_z_history.astype(jnp.float32)
        done = False
        return obs, new_state, reward, done, {}

    @property
    def num_actions(self):
        return 2

    def action_space(self, params: LimitedMemoryEnvParams):
        return spaces.Discrete(2)

    def observation_space(self, params: LimitedMemoryEnvParams):
        return spaces.Box(
            low=0, high=1, shape=(params.window_size,), dtype=jnp.float32
        )

    def state_space(self, params: LimitedMemoryEnvParams):
        return spaces.Box(low=0, high=jnp.inf, shape=(), dtype=jnp.int32)

    def reset_env(self, key: chex.PRNGKey, params: LimitedMemoryEnvParams):
        initial_z_history = jnp.zeros(params.window_size, dtype=jnp.int32)
        state = LimitedMemoryEnvState(time=0, z_history=initial_z_history)
        obs = initial_z_history.astype(jnp.float32)
        return obs, state

    @property
    def default_params(self):
        return LimitedMemoryEnvParams(
            reward_tensor=jnp.array([[0.0, 0.1], [0.2, 0.3]]), noise_std=1.0
        )
