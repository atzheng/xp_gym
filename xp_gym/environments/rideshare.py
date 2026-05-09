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
from or_gymnax.rideshare_pool import (
    ManhattanRidesharePoolDispatch,
    GreedyPolicy as PoolGreedyPolicy,
)

from xp_gym.environments.environment import XPEnvironment, XPEnvParams


class XPRidesharePricingEnv(XPEnvironment):
    def __init__(
        self,
        n_cars: int = 300,
        n_events: int = 500000,
        price_per_distance_A: float = 0.01,
        price_per_distance_B: float = 0.02,
        **kwargs
    ):
        super().__init__(
            env=ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events, **kwargs),
            policy_A=SimplePricingPolicy(
                n_cars=n_cars, price_per_distance=price_per_distance_A
            ),
            policy_B=SimplePricingPolicy(
                n_cars=n_cars, price_per_distance=price_per_distance_B
            ),
        )


class XPRidesharePoolDispatchEnv(XPEnvironment):
    def __init__(
        self,
        n_cars: int = 300,
        n_events: int = 500000,
        savings_threshold_A: float = 0.0,
        savings_threshold_B: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__(
            env=ManhattanRidesharePoolDispatch(n_cars=n_cars, n_events=n_events),
            policy_A=PoolGreedyPolicy(
                n_cars=n_cars,
                temperature=temperature,
                savings_threshold=savings_threshold_A,
            ),
            policy_B=PoolGreedyPolicy(
                n_cars=n_cars,
                temperature=temperature,
                savings_threshold=savings_threshold_B,
            ),
        )
