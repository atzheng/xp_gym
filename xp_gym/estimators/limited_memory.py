from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import Environment, EnvParams
from jaxtyping import Float, Integer, Bool
from typing import Tuple, Any
from jax import Array
import jax.numpy as jnp
import jax
from dataclasses import field

from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.observation import Observation
from xp_gym.designs.design import Design
from or_gymnax.rideshare import obs_to_state
from jaxtyping import Bool, Float, Integer


def get_dummy_obs(
    design: Design,
    env: Environment,
    env_params: EnvParams,
):
    """
    Get a dummy observation from the environment.
    Use for initializing structs that require an Observation.
    This will return an object with the correct structure / shapes, but
    no guarantee of having semantically meaningful content.
    """
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)
    action, design_info = design.assign_treatment(
        design.reset(rng, env_params), state
    )
    obs, state, reward, _, info = env.step(rng, state, action, env_params)
    return Observation(
        obs=obs,
        action=action,
        reward=reward,
        info=info,
        design_info=design_info,
    )


@struct.dataclass
class LimitedMemoryEstimatorState(EstimatorState):
    """
    State for a limited memory network estimator.

    @param t: current time step, used also for computing the window pointer
    @param obs: observations stored in the limited memory
    @param estimator_state: arbitrary additional state for the estimator
    """

    t: int
    obs: Observation
    estimator_state: Any


@struct.dataclass
class LimitedMemoryEstimator(Estimator):
    """
    Base class for an estimator that stores a limited window of information
    (e.g., for computing interference.)
    """

    window_size: int

    def process_obs(
        self,
        design,
        env: Environment,
        env_params: EnvParams,
        obs: Observation,
    ) -> Observation:
        """
        Optional preprocessing of an observation to e.g., extract relevant features,
        reduce memory consumption. Does nothing by default.
        """
        return obs

    def reset(self, rng, env, env_params, design):
        dummy_obs = self.process_obs(
            design, env, env_params, get_dummy_obs(design, env, env_params)
        )
        return LimitedMemoryEstimatorState(
            t=0,
            obs=jax.tree.map(
                lambda x: jnp.repeat(
                    jnp.expand_dims(x, 0), self.window_size, axis=0
                ),
                dummy_obs,
            ),
            # A float by default, but can use any arbitrary data structure by
            # overriding reset
            estimator_state=0.0,
        )

    def update(
        self,
        env: Environment,
        env_params: EnvParams,
        design: Design,
        state: LimitedMemoryEstimatorState,
        obs: Observation,
    ):
        window_ptr = state.t % self.window_size
        new_obs = jax.tree.map(
            lambda a, b: a.at[window_ptr].set(b),
            state.obs,
            self.process_obs(design, env, env_params, obs),
        )
        return LimitedMemoryEstimatorState(
            t=state.t + 1, obs=new_obs, estimator_state=state.estimator_state
        )
