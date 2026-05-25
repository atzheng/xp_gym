"""
Pooled ridesharing dispatch environment for experiments.
Imports core implementation from or_gymnax and adds the XP wrapper.
"""
from typing import Union
from flax import struct
from jaxtyping import Float
from jax import Array

from or_gymnax.rideshare_pool import (
    EnvState,
    EnvParams,
    RidesharePoolDispatch,
    ManhattanRidesharePoolDispatch,
    GreedyPolicy,
    obs_to_state,
    insert_and_optimize_trip,
    optimize_waypoints,
    _num_wp,
    admissible_sequences,
    get_sequences,
    compute_real_car_costs,
    greedy_select_car,
    check_ghost_triggers,
    update_triggered_ghosts,
    create_ghost_pair,
    write_ghosts_to_buffer,
    expire_ghosts,
)

from xp_gym.environments.environment import XPEnvironment


class XPRidesharePoolDispatchEnv(XPEnvironment):
    def __init__(
        self,
        n_cars: int = 300,
        n_events: int = 500000,
        savings_threshold_A: float = 0.0,
        savings_threshold_B: float = 0.2,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(
            env=ManhattanRidesharePoolDispatch(
                n_cars=n_cars, n_events=n_events, **kwargs
            ),
            policy_A=GreedyPolicy(
                n_cars=n_cars,
                temperature=temperature,
                savings_threshold=savings_threshold_A,
            ),
            policy_B=GreedyPolicy(
                n_cars=n_cars,
                temperature=temperature,
                savings_threshold=savings_threshold_B,
            ),
        )
