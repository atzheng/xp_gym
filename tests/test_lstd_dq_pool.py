"""Online PoolLSTDDQEstimator == offline numpy recomputation on the same trajectory."""
import numpy as np
import jax, jax.numpy as jnp
from scipy.signal import lfilter

from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.designs.design import UnitRandomizedDesign
from xp_gym.estimators.lstd_dq_pool import PoolLSTDDQEstimator
from xp_gym.simulator import init_carry
from xp_gym.observation import Observation
from or_gymnax.rideshare_pool import obs_to_state, insert_and_optimize_trip

CH, NCH = 500, 4


def main():
    env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_B=0.1)
    p0 = env.default_params
    params = p0.replace(env_params=p0.env_params.replace(
        max_active_trips=2, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))
    est = PoolLSTDDQEstimator(gamma=0.999, ridge=1e-3, trace_lambdas=(0.0, 0.9), chunk=CH,
                              use_drop=True, match_K=8, match_offsets=(0, 120), match_K_later=4,
                              jackknife_blocks=2, jackknife_block_chunks=2)
    design = UnitRandomizedDesign(p=0.5)
    ests = {"lstd": est}
    carry = init_carry(ests, design, env, params, jax.random.PRNGKey(0))

    @jax.jit
    def run_chunk(carry, key):
        def f(c, k):
            obs, state, est_states, ds = c
            action, info_d = design.assign_treatment(ds, state)
            new_obs, new_state, reward, _, info = env.step(k, state, action, params)
            xo = Observation(obs=new_obs, action=action, reward=reward, info=info, design_info=info_d)
            es = {"lstd": est.update(env, params, design, est_states["lstd"], xo)}
            c2 = (new_obs, new_state, es, design.update(ds, xo))
            return c2, (obs, new_obs, action, reward, info["action_A"], info["action_B"])
        carry, (pre, post, *rest) = jax.lax.scan(f, carry, jax.random.split(key, CH))
        pre = (pre, rest)
        st = est.end_chunk(env, params, design, carry[2]["lstd"])
        carry = (carry[0], carry[1], {"lstd": st}, carry[3])
        return carry, (pre, post), st

    logs = []
    for c in range(NCH):
        carry, (pre, post), st = run_chunk(carry, jax.random.PRNGKey(100 + c))
        logs.append((jax.tree.map(np.asarray, pre), np.asarray(post), st))
    online = np.asarray(est.estimate(env, params, design, carry[2]["lstd"]))

    # ---- offline recomputation from the raw trajectory ----
    st = carry[2]["lstd"]
    setup = est._setup(params)
    ip = params.env_params
    pre = np.concatenate([l[0][0] for l in logs]); post = np.concatenate([l[1] for l in logs])
    z, r, cA, cB = [np.concatenate([l[0][1][i] for l in logs]).astype(np.float64) for i in range(4)]
    # recover z, r, cars by replaying (the estimator does not store them; recompute)
    print("online", online)
    # Check LSTD stats consistency: recompute phi rows and stats with numpy
    feats = jax.jit(jax.vmap(lambda o: est._features(setup, *obs_to_state(300, 4, o)[1:],
                                                     obs_to_state(300, 4, o)[0].t)[0]))
    Phi = np.asarray(feats(jnp.asarray(post)), dtype=np.float64)
    X0, X1 = Phi[:-1], Phi[1:]
    Cd = X0.T @ (X0 - X1)
    rel = np.abs(np.asarray(st.Cd) - Cd).max() / np.abs(Cd).max()
    print("Cd rel err", rel)
    assert rel < 1e-4
    assert np.isfinite(online).all()

    # paired / trace accumulators
    def cf_feats(o_pre, o_post, ca, cb):
        ev, Wp, Tp = obs_to_state(300, 4, o_pre)
        evn, _, _ = obs_to_state(300, 4, o_post)
        cbi = jnp.maximum(cb, 0).astype(jnp.int32)
        wp, t, mc, _ = insert_and_optimize_trip(ip.distances, Wp[cbi], Tp[cbi], ev.src, ev.dest, ev.t, 2)
        W = jnp.where(cb >= 0, Wp.at[cbi].set(wp), Wp); T = jnp.where(cb >= 0, Tp.at[cbi].set(t), Tp)
        rcf = jnp.where(cb >= 0, ip.distances[ev.src, ev.dest] * 2.0 - mc, 0.0)
        return est._features(setup, W, T, evn.t)[0], rcf
    Pcf, rcf = jax.jit(jax.vmap(cf_feats))(jnp.asarray(pre), jnp.asarray(post),
                                            jnp.asarray(cA, jnp.int32), jnp.asarray(cB, jnp.int32))
    Pcf, rcf = np.asarray(Pcf, np.float64), np.asarray(rcf, np.float64)
    s_ = 2 * z - 1
    d = (cA != cB).astype(float); d[0] = 0
    P_r = np.sum(s_ * d * (r - rcf)); P_phi = (s_ * d)[:, None] * (Pcf - Phi)
    print("P_r", P_r, float(st.P_r))
    assert abs(P_r - float(st.P_r)) < 1e-3 * max(1, abs(P_r))
    assert np.allclose(P_phi.sum(0), np.asarray(st.P_phi), rtol=1e-3, atol=1e-2)
    w = 2 * s_ * d
    for j, lam in enumerate(est.trace_lambdas):
        e = lfilter([1.0], [1.0, -lam * est.gamma], w)
        T_r = np.sum(e[1:] * r[1:]); T_phi = (e[1:, None] * (est.gamma * Phi[1:] - Phi[:-1])).sum(0)
        print("T_r", lam, T_r, float(st.T_r[j]))
        assert abs(T_r - float(st.T_r[j])) < 1e-3 * max(1, abs(T_r))
        assert np.allclose(T_phi, np.asarray(st.T_phi[j]), rtol=1e-3, atol=1e-1)
    # jackknife: block stats sum to the full stats; jk combination by hand
    for k in ["C", "Cd", "b1", "n"]:
        assert np.allclose(np.asarray(st.blk[k]).sum(0), np.asarray(getattr(st, k)), rtol=1e-4, atol=1e-2)
    L = 1 + len(est.trace_lambdas)
    assert online.shape == (2 * L,)

    def np_theta(Cd, C, b1, s1, s2, sr, n):
        Cd, C, b1, s1, s2 = (np.asarray(x, np.float64) for x in (Cd, C, b1, s1, s2))
        n = float(n); rbar = float(sr) / n
        sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)); sd = np.where(sd > 1e-6, sd, 1.0); sd[-1] = 1
        A = (Cd + (1 - est.gamma) * C) / n / np.outer(sd, sd); b = (b1 - s1 * rbar) / n / sd
        return np.linalg.solve(A + est.ridge * np.eye(len(A)), b) / sd, rbar
    keys = ["Cd", "C", "b1", "s1", "s2", "sr", "n"]
    th, rbar = np_theta(*[getattr(st, k) for k in keys])
    T = float(st.t)
    taus = lambda th: np.concatenate([[(float(st.P_r) - est.gamma * th @ np.asarray(st.P_phi)) / T],
                                      (np.asarray(st.T_r) - rbar * np.asarray(st.T_e) + np.asarray(st.T_phi) @ th) / T])
    loo = [taus(np_theta(*[np.asarray(getattr(st, k)) - np.asarray(st.blk[k])[b] for k in keys])[0]) for b in range(2)]
    jk = 2 * taus(th) - np.mean(loo, 0)
    print("jk", jk, online[L:])
    assert np.allclose(jk, online[L:], rtol=2e-2, atol=0.5)
    print("ok")


if __name__ == "__main__":
    main()
