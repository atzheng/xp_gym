from flax import struct
from chex import PRNGKey
from gymnax.environments.environment import EnvParams, Environment
from xp_gym.observation import Observation


@struct.dataclass
class EstimatorState:
    """
    Abstract base class for estimator states
    """
    pass



@struct.dataclass
class Estimator:
    """
    Abstract base class for estimators.
    """
    def reset(self, rng: PRNGKey, env: Environment, env_params: EnvParams, design) -> EstimatorState:
        """
        Initialize the estimator with necessary parameters.
        """
        raise NotImplementedError("Initialize method must be implemented in subclass.")

    def update(self, env: Environment, env_params: EnvParams, design, state: EstimatorState, obs: Observation):
        """
        Update the estimator with new data.
        """
        raise NotImplementedError("Update method must be implemented in subclass.")

    def estimate(self, env: Environment, env_params: EnvParams, design, state: EstimatorState):
        """
        Estimate the value based on the current state of the estimator.
        """
        raise NotImplementedError("Estimate method must be implemented in subclass.")
