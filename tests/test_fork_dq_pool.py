"""ForkDQEstimator's fork accumulators == brute-force forked env rollouts.

For every differing step t0 of a short p=0.5 trajectory, re-run the env from the
pre-step state with the arm flipped at t0 and the same keys / arms afterwards,
and compare sum_t s_t [sum_{j<H} (r_real - r_fork)] and sum_t s_t [phi(fork) -
phi(real)] at age H with the estimator's acc_r / acc_phi.
"""
import numpy as np
import jax, jax.numpy as jnp

from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.designs.design import UnitRandomizedDesign
from xp_gym.estimators.fork_dq_pool import ForkDQEstimator
from xp_gym.simulator import init_carry
from xp_gym.observation import Observation
from or_gymnax.rideshare_pool import obs_to_state

N = 300
HS = (1, 5, 20)
STA, STB = 0.0, 0.3


def main():
    env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_A=STA,
                                     savings_threshold_B=STB)
    p0 = env.default_params
    params = p0.replace(env_params=p0.env_params.replace(
        max_active_trips=2, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))
    est = ForkDQEstimator(gamma=1.0, ridge=1e-6, trace_lambdas=(0.0,), chunk=N, thresholds=(STA, STB),
                          fork_horizons=HS, fork_slots=64, fork_ghosts=40)
    design = UnitRandomizedDesign(p=0.5)
    carry = init_carry({"f": est}, design, env, params, jax.random.PRNGKey(3))
    keys = jax.random.split(jax.random.PRNGKey(7), N)

    @jax.jit
    def run(carry, keys):
        def f(c, k):
            obs, state, es, ds = c
            action, info_d = design.assign_treatment(ds, state)
            new_obs, new_state, reward, _, info = env.step(k, state, action, params)
            xo = Observation(obs=new_obs, action=action, reward=reward, info=info, design_info=info_d)
            c2 = (new_obs, new_state, {"f": est.update(env, params, design, es["f"], xo)}, design.update(ds, xo))
            return c2, (state, action, reward, info["action_A"], info["action_B"], new_obs)
        return jax.lax.scan(f, carry, keys)

    carry, (states, z, r, cA, cB, obs_post) = run(carry, keys)
    st = carry[2]["f"]
    z, r, cA, cB = (np.asarray(x) for x in (z, r, cA, cB))
    setup = est._setup(params)

    @jax.jit
    def rollout(s0, keys, zs):
        def f(s, kz):
            k, zz = kz
            o, s2, rr, _, _ = env.step(k, s, zz, params)
            return s2, (rr, o)
        _, (rr, oo) = jax.lax.scan(f, s0, (keys, zs))
        return rr, oo

    phi = jax.jit(lambda o: est._features(setup, *obs_to_state(300, 4, o)[1:], obs_to_state(300, 4, o)[0].t)[0])
    Hmax = max(HS)
    exp_r = np.zeros(len(HS)); exp_phi = np.zeros((len(HS), len(st.prev_phi)))
    nf = 0
    for t0 in range(1, N):
        if cA[t0] == cB[t0] or t0 + 1 > N - 1:
            continue
        L = min(Hmax, N - t0)
        s0 = jax.tree.map(lambda x: x[t0], states)
        zs = jnp.asarray(z[t0:t0 + L]).at[0].set(1 - z[t0])
        rf, of = rollout(s0, keys[t0:t0 + L], zs)
        rf = np.asarray(rf)
        s = 2.0 * z[t0] - 1
        for i, H in enumerate(HS):
            if t0 + H > N - 1:      # checkpoint processed at step t0+H
                continue
            cum = np.sum(r[t0:t0 + H] - rf[:H])
            exp_r[i] += s * cum
            dphi = np.asarray(phi(of[H - 1])) - np.asarray(phi(obs_post[t0 + H - 1]))
            dphi[-1] = 0
            exp_phi[i] += s * dphi
        nf += 1
    print("forks", nf, "early", float(st.n_early), "drop", float(st.n_drop))
    print("acc_r  ", np.asarray(st.acc_r))
    print("brute  ", exp_r)
    assert float(st.n_early) == 0 and float(st.n_drop) == 0
    assert np.allclose(np.asarray(st.acc_r), exp_r, rtol=1e-4, atol=1e-2)
    err = np.abs(np.asarray(st.acc_phi) - exp_phi).max()
    print("acc_phi max err", err)
    assert err < 1e-3
    # H=1 checkpoint equals the paired accumulators
    assert np.allclose(float(st.acc_r[0]), float(st.P_r), rtol=1e-4, atol=1e-2)
    out = np.asarray(est.estimate(env, params, design, st))
    print(dict(zip(est.labels, out.round(3))))
    print("ok")


if __name__ == "__main__":
    main()
