"""
Pooled ridesharing dispatch environment for experiments.
Imports core implementation from or_gymnax and adds the XP wrapper.
"""
from typing import Union
from flax import struct
from jaxtyping import Float
from jax import Array

import jax
import jax.numpy as jnp

from or_gymnax import rideshare as rs
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


# ── Synthetic 3-node scenario (ghost_sim.py demand) ──────────────────────────

_SYNTHETIC_DISTANCES = jnp.array(
    [[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=jnp.int32
) * 100


def _make_synthetic_events(n_events: int) -> rs.RideshareEvent:
    k1, _ = jax.random.split(jax.random.PRNGKey(1), 2)
    times = jnp.cumsum(jax.random.randint(k1, (n_events,), 1, 5))
    srcs = jnp.zeros(n_events, dtype=jnp.int32)
    dests = (1 + jnp.arange(n_events, dtype=jnp.int32)) % 2 + 1
    return rs.RideshareEvent(t=times, src=srcs, dest=dests)


class SyntheticRidesharePoolDispatch(RidesharePoolDispatch):
    """
    Fixed 3-node, synthetic-demand rideshare environment matching
    or-gymnax/scripts/ghost_sim.py.

    Events are generated once from PRNGKey(1) with monotone increasing
    times, all pickups at node 0, alternating destinations (nodes 1/2).
    Distance matrix: d(0,1)=100, d(1,2)=100, d(0,2)=200.
    Cars are initialised with waypoints[:, 0]=1 on reset.
    """

    N_NODES = 3

    def __init__(self, n_cars: int = 1, n_events: int = 30):
        super().__init__(n_cars=n_cars, n_nodes=self.N_NODES, n_events=n_events)

    @property
    def name(self) -> str:
        return "SyntheticRidesharePoolDispatch-v0"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            events=_make_synthetic_events(self.n_events),
            distances=_SYNTHETIC_DISTANCES,
            n_cars=self.n_cars,
            max_active_trips=2,
            max_ghosts=2 * self.N_NODES,
            ghost_max_lifespan=50,
        )

    def reset_env(self, key, params: EnvParams):
        obs, state = super().reset_env(key, params)
        state = state.replace(waypoints=state.waypoints.at[:, 0].set(1))
        return self.get_obs(state), state


class XPSyntheticPoolEnv(XPEnvironment):
    """
    XPEnvironment backed by SyntheticRidesharePoolDispatch.
    Drop-in replacement for XPRidesharePoolDispatchEnv for smoke tests and
    debugging without loading Manhattan data.
    """

    def __init__(
        self,
        n_cars: int = 1,
        n_events: int = 30,
        savings_threshold_A: float = 0.0,
        savings_threshold_B: float = 0.6,
        temperature: float = 0.1,
    ):
        super().__init__(
            env=SyntheticRidesharePoolDispatch(n_cars=n_cars, n_events=n_events),
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
