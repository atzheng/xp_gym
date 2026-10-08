"""TSR design / env / TSRI estimators."""
from functools import lru_cache
import math

import numpy as np
import jax
import jax.numpy as jnp

from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.environments.rideshare_pool_tsr import XPRidesharePoolTSREnv
from xp_gym.designs.tsr import TwoSidedRandomizedDesign, paper_tsr_params
from xp_gym.estimators.tsri import TSRIEstimator, tsr_estimates
from xp_gym.simulator import init_carry
from xp_gym.observation import Observation
from tests.pool_env_params import pool_params

N = 300


@lru_cache(None)
def envs():
    uenv = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_B=0.2)
    tenv = XPRidesharePoolTSREnv(n_cars=300, n_events=500000, savings_threshold_B=0.2)
    return [(uenv, pool_params()), (tenv, pool_params())]


def rollout(env, params, action_fn, n=N, seed=0):
    @jax.jit
    def go(key):
        obs, state = env.reset(key, params)

        def f(c, k):
            obs, state = c
            o, s, r, _, info = env.step(k, state, action_fn(state), params)
            return (o, s), (r, info)
        return jax.lax.scan(f, (obs, state), jax.random.split(key, n))[1]
    return jax.tree.map(np.asarray, go(jax.random.PRNGKey(seed)))


def test_tsr_env_reproduces_global_arms():
    """All cars treated == unit env's arm B; none treated == arm A; car_A/car_B
    are the arms' choices."""
    (uenv, up), (tenv, tp) = envs()
    for z in (0, 1):
        ru, iu = rollout(uenv, up, lambda s: jnp.bool_(z))
        rt, it = rollout(tenv, tp, lambda s: jnp.full(300, bool(z)))
        assert np.array_equal(ru, rt)
        assert np.array_equal(iu["action_A"], it["car"])          # dispatched car
        assert np.array_equal(iu["action_A"], it["car_B" if z else "car_A"])
        assert np.array_equal(iu["action_B"], it["car_A" if z else "car_B"])
    assert (it["car_A"] != it["car_B"]).mean() > 0.05


def test_tsr_env_mixed_rule():
    """Per-car thresholds: replay the dispatch rule in numpy from the logged
    pre-decision states (car j eligible iff feasible and solo or
    cost < direct * (1 - threshold_j), threshold_j = B's iff treated)."""
    from or_gymnax.rideshare_pool import compute_real_car_costs
    (_, _), (tenv, tp) = envs()
    mask = jnp.arange(300) % 2 == 0
    ip = tp.env_params

    @jax.jit
    def go(key):
        obs, state = tenv.reset(key, tp)

        def f(c, k):
            o, s, r, _, info = tenv.step(k, c, mask, tp)
            costs, feas = compute_real_car_costs(ip.distances, c.waypoints, c.times, c.event, 2)
            solo = jnp.all(c.times <= c.event.t, axis=1)
            return s, (info["car"], costs, feas, solo, ip.distances[c.event.src, c.event.dest])
        return jax.lax.scan(f, state, jax.random.split(key, N))[1]
    car, costs, feas, solo, direct = jax.tree.map(np.asarray, go(jax.random.PRNGKey(3)))
    th = np.where(np.asarray(mask), 0.2, 0.0)
    n_mixed = 0
    for t in range(N):
        elig = feas[t] & (solo[t] | (costs[t] < direct[t] * (1 - th)))
        ref = np.argmin(np.where(elig, costs[t], np.iinfo(costs.dtype).max)) if elig.any() else -1
        assert car[t] == ref
        eA = feas[t] & (solo[t] | (costs[t] < direct[t]))
        n_mixed += int(ref != (np.argmin(np.where(eA, costs[t], np.iinfo(costs.dtype).max))
                               if eA.any() else -1))
    assert n_mixed > 0   # the treated cars' threshold mattered at some steps


def test_design_and_paper_params():
    aC, aL, b = paper_tsr_params(0.5, 0.5, 1.0)
    e = math.exp(-1)
    assert np.isclose(aC, (1 - e) + 0.5 * e) and np.isclose(aL, 0.5 * (1 - e) + e)
    assert np.isclose(b, e)
    (_, _), (tenv, tp) = envs()
    d = TwoSidedRandomizedDesign(a_C=0.3, a_L=0.6)
    ds = d.reset(jax.random.PRNGKey(0), tp)
    _, st = tenv.reset(jax.random.PRNGKey(1), tp)
    zs = []
    for t in range(400):
        a, info = d.assign_treatment(ds, st.replace(time=t))
        assert np.array_equal(np.asarray(info.z_L), np.asarray(ds.z_L))
        assert np.array_equal(np.asarray(a), np.asarray(info.z_C & ds.z_L))
        zs.append(bool(info.z_C))
    assert abs(np.mean(zs) - 0.3) < 0.08
    assert abs(np.asarray(ds.z_L).mean() - 0.6) < 0.1


def test_tsri_accumulators_and_formulas():
    (_, _), (env, params) = envs()
    design = TwoSidedRandomizedDesign(a_C=0.5, a_L=0.4)
    est = TSRIEstimator(betas=(0.0, 0.37, 1.0), ks=(1, 2))
    carry = init_carry({"e": est}, design, env, params, jax.random.PRNGKey(5))

    @jax.jit
    def go(carry, key):
        def f(c, k):
            obs, state, es, ds = c
            a, di = design.assign_treatment(ds, state)
            o, s, r, _, info = env.step(k, state, a, params)
            xo = Observation(obs=o, action=a, reward=r, info=info, design_info=di)
            return ((o, s, {"e": est.update(env, params, design, es["e"], xo)},
                     design.update(ds, xo)), (r, info["car"], di.z_C))
        return jax.lax.scan(f, carry, jax.random.split(key, N))
    carry, (r, car, zC) = jax.tree.map(np.asarray, go(carry, jax.random.PRNGKey(6)))
    zL = np.asarray(carry[3].z_L)
    Q = np.zeros((2, 2))
    for t in range(N):
        if car[t] >= 0:
            Q[int(zC[t]), int(zL[car[t]])] += r[t]
    st = carry[2]["e"]
    assert np.allclose(Q, np.asarray(st.Q), rtol=1e-5)
    out = np.asarray(est.estimate(env, params, design, st))
    lab = dict(zip(est.labels, out))
    aC, aL = 0.5, 0.4
    q = Q / N
    q11, q01 = q[1, 1] / (aC * aL), q[0, 1] / ((1 - aC) * aL)
    q10, q00 = q[1, 0] / (aC * (1 - aL)), q[0, 0] / ((1 - aC) * (1 - aL))
    assert np.isclose(lab["tsrn"], q11 - (q[0, 1] + q[1, 0] + q[0, 0]) / (1 - aC * aL), rtol=1e-4)
    for k in (1, 2):
        for b in (0.0, 0.37, 1.0):
            ref = (b * (q11 - q01 - k * (1 - b) * (q00 - q01))
                   + (1 - b) * (q11 - q10 - k * b * (q00 - q10)))     # eq. (28)
            assert np.isclose(lab[f"tsri{k}_b{b:g}"], ref, rtol=1e-4, atol=1e-3)
        assert np.isclose(lab[f"tsri{k}_b1"], lab["cr"], rtol=1e-5)   # beta=1 -> CR
        assert np.isclose(lab[f"tsri{k}_b0"], lab["lr"], rtol=1e-5)   # beta=0 -> LR
