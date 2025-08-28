from flax import struct
from or_gymnax.rideshare import (
    ManhattanRideshareDispatch,
    ManhattanRidesharePricing,
    GreedyPolicy,
    SimplePricingPolicy,
    EnvParams,
    obs_to_state,
    RideshareEvent,
)

from xp_gym.environments.environment import XPEnvironment, XPEnvParams


class XPRidesharePricingEnv(XPEnvironment):
    def __init__(
        self,
        n_cars: int = 300,
        price_per_distance_A: float = 0.01,
        price_per_distance_B: float = 0.02,
        **kwargs
    ):
        super().__init__(
            env=ManhattanRidesharePricing(n_cars=n_cars, **kwargs),
            policy_A=SimplePricingPolicy(
                n_cars=n_cars, price_per_distance=price_per_distance_A
            ),
            policy_B=SimplePricingPolicy(
                n_cars=n_cars, price_per_distance=price_per_distance_B
            ),
        )
