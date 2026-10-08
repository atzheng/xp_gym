"""Shared EnvParams for tests: jax caches compare pytree aux data, and two
separately loaded EnvParams hold distinct static arrays, so every test in a
process must reuse the same params object."""
from functools import lru_cache

from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv


@lru_cache(None)
def pool_params():
    p0 = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000).default_params
    return p0.replace(env_params=p0.env_params.replace(
        max_active_trips=2, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))
