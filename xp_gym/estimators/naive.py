from flax import struct
import jax.numpy as jnp

from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.observation import Observation


@struct.dataclass
class NaiveEstimatorState(EstimatorState):
    estimate: float  # Cumulative inverse probability weighting
    count: int


@struct.dataclass
class NaiveEstimator(Estimator):
    """
    Naive IPW estimator that computes the average outcome for treated and control groups,
    ignoring interference.

    @param use_known_actions: If True, assumes obs.info has "action_A" and "action_B"
    fields. When action_A == action_B, the estimate is not updated but count still
    increments (the observation provides no counterfactual information).
    """

    use_known_actions: bool = False

    def reset(self, rng, env, env_params, design):
        return NaiveEstimatorState(0.0, 0)

    def update(self, env, env_params, design, state: NaiveEstimatorState, obs: Observation):
        delta = (
            obs.reward * obs.action / obs.design_info.p
            - obs.reward * (1 - obs.action) / (1 - obs.design_info.p)
        )
        if self.use_known_actions:
            action_A = obs.info["action_A"].reshape(-1)[0]
            action_B = obs.info["action_B"].reshape(-1)[0]
            delta = jnp.where(action_A != action_B, delta, 0.0)
        return NaiveEstimatorState(state.estimate + delta, state.count + 1)

    def estimate(self, env, env_params, design, state: NaiveEstimatorState):
        return state.estimate / state.count
