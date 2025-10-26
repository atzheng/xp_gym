from typing import Callable, Dict, Tuple, Any, Optional
import jax
from jax import Array
import chex
from jaxtyping import Integer
from jax import numpy as jnp

from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.designs.design import Design, DesignState
from xp_gym.observation import Observation
import gymnax
from gymnax.environments.environment import EnvState, Environment, EnvParams
from jax_tqdm import scan_tqdm


def step(
    estimators: Dict[str, Estimator],
    design: Design,
    env: Environment,
    env_params: EnvParams,
    carry: Tuple[Array, Array, Dict[str, EstimatorState], DesignState],
    rng: Any,
):
    """
    Single step of the environment, updating the estimators and design.
    """
    obs, state, est_states, design_state = carry
    action, design_info = design.assign_treatment(design_state, state)
    new_obs, new_state, reward, _, _ = env.step(rng, state, action, env_params)
    new_xp_obs = Observation(
        obs=new_obs,
        action=action,
        reward=reward,
        design_info=design_info,
    )
    new_est_states = {
        est_name: estimator.update(
            env, env_params, est_states[est_name], new_xp_obs
        )
        for est_name, estimator in estimators.items()
    }
    new_design_state = design.update(design_state, new_xp_obs)
    return ((new_obs, new_state, new_est_states, new_design_state), None)


def step_n_and_estimate(
    estimators: Dict[str, Estimator],
    design: Design,
    env: Environment,
    env_params: EnvParams,
    carry: Tuple[Array, Array, Dict[str, EstimatorState], DesignState],
    rng: Any,
    n: Integer,
):
    """
    Step the environment n times and return the current estimates from each estimator.
    """
    carry, _ = jax.lax.scan(
        jax.tree_util.Partial(step, estimators, design, env, env_params),
        carry,
        jax.random.split(rng, n),
    )
    est_states = carry[2]
    estimates = {
        est_name: estimator.estimate(env, env_params, est_states[est_name])
        for est_name, estimator in estimators.items()
    }
    return carry, estimates


def simulate(
    estimators: Dict[str, Estimator],
    design: Design,
    env: Environment,
    env_params: EnvParams,
    rng: chex.PRNGKey,
    T: Optional[int] = None,
    estimate_every_n_steps: Optional[int] = None,
):
    """
    Simulate an environment with given estimators and design for T steps.
    """
    rng, reset_rng = jax.random.split(rng)
    rng, estimator_rng = jax.random.split(rng)
    rng, design_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    init_est_states = {
        est_name: estimator.reset(estimator_rng, env, env_params)
        for est_name, estimator in estimators.items()
    }
    init_carry = (
        obs,
        state,
        init_est_states,
        design.reset(design_rng, env_params),
    )

    T = T or env_params.max_steps_in_episode
    estimate_every_n_steps = estimate_every_n_steps or T
    n_estimates = T // estimate_every_n_steps
    rngs = jax.random.split(rng, n_estimates)

    @scan_tqdm(n_estimates, print_rate=10)
    def scanner(carry, idx_and_rng):
        idx, rng = idx_and_rng
        return step_n_and_estimate(
            estimators,
            design,
            env,
            env_params,
            carry,
            rng,
            n=estimate_every_n_steps,
        )

    carry, results = jax.lax.scan(
        scanner, init_carry, (jnp.arange(n_estimates), rngs)
    )
    return carry, results
