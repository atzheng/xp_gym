from flax import struct

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
    ignoring interference
    """

    def reset(self, rng, env, env_params, design):
        return NaiveEstimatorState(0.0, 0)

    def update(self, env, env_params, design, state: NaiveEstimatorState, obs: Observation):
        return NaiveEstimatorState(
            state.estimate
            + obs.reward * obs.action / obs.design_info.p
            - obs.reward * (1 - obs.action) / (1 - obs.design_info.p),
            state.count + 1,
        )

    def estimate(self, env, env_params, design, state: NaiveEstimatorState):
        return state.estimate / state.count
