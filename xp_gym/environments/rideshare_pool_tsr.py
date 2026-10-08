"""Pooled dispatch env with per-car treatment, for two-sided randomization.

The action is a bool vector (n_cars,): car j is screened for the current
request with arm B's pooling savings threshold if action[j], else with arm
A's.  The dispatcher then picks the cheapest eligible car, exactly as
or_gymnax.rideshare_pool.greedy_select_car does with a scalar threshold
(eligible = feasible & (solo | cost < direct_cost * (1 - threshold))).
With action all-False (all-True) this reproduces the unit env's arm A (B)
step exactly (same keys, same car).

info adds
  car    the dispatched car (-1 if unfulfilled),
  car_A  the car the all-A policy would pick (-1 if none),
  car_B  the car the all-B policy would pick (-1 if none).
Ghost groups are never created (the ghost bookkeeping of the inner env runs
but stays empty), so keep max_groups etc. at 1 for speed.
"""
import jax
import jax.numpy as jnp

from or_gymnax.rideshare_pool import compute_real_car_costs
from xp_gym.environments.rideshare_pool import XPRidesharePoolDispatchEnv


class XPRidesharePoolTSREnv(XPRidesharePoolDispatchEnv):
    def step_env(self, key, state, treat_cars, params):
        key, policy_key = jax.random.split(key, 2)
        key, step_key = jax.random.split(key, 2)
        ip = params.env_params
        inner = self.env
        obs = inner.get_obs(state, ip)
        thA = self.policy_A.apply(ip, dict(), obs, policy_key, **params.policy_A_kwargs)[0][0]
        thB = self.policy_B.apply(ip, dict(), obs, policy_key, **params.policy_B_kwargs)[0][0]
        thA, thB = thA.astype(jnp.float32), thB.astype(jnp.float32)

        ev = state.event
        costs, feas = compute_real_car_costs(
            ip.distances, state.waypoints, state.times, ev, ip.max_active_trips)
        solo = jnp.all(state.times <= ev.t, axis=1)
        direct = ip.distances[ev.src, ev.dest]
        maxint = jnp.iinfo(costs.dtype).max

        def pick(th):
            elig = feas & (solo | (costs < direct * (1 - th)))
            car = jnp.argmin(jnp.where(elig, costs, maxint))
            return jnp.where(elig.any(), car, -1).astype(jnp.int32)

        car = pick(jnp.where(jnp.asarray(treat_cars), thB, thA))
        car_A, car_B = pick(thA), pick(thB)
        next_obs, next_state, reward, done, info = jax.lax.cond(
            car >= 0,
            lambda: inner.step_env_dispatch(step_key, state, jnp.maximum(car, 0), ip, thA),
            lambda: inner.step_env_unfulfill(step_key, state, ip, thA),
        )
        return next_obs, next_state, reward, done, {**info, "car": car, "car_A": car_A,
                                                    "car_B": car_B}
