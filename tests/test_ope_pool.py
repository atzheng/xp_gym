"""PoolOPEEstimator: online accumulators == straightforward numpy replay of the
same trajectory, and on-policy sanity checks."""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.designs.design import UnitRandomizedDesign
from xp_gym.estimators.ope_pool import PoolOPEEstimator, np_rhos, np_estimate
from xp_gym.simulator import init_carry
from xp_gym.observation import Observation
from tests.pool_env_params import pool_params

CH, NCH = 400, 4


@lru_cache(None)   # one env instance: jitted env methods use `self` as a static arg
def make_env():
    env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_B=0.1)
    return env, pool_params()


def run(p, est, nch=NCH, seed=0, ch=CH):
    env, params = make_env()
    design = UnitRandomizedDesign(p=p)
    carry = init_carry({"e": est}, design, env, params, jax.random.PRNGKey(seed))

    @jax.jit
    def run_chunk(carry, key):
        def f(c, k):
            obs, state, es, ds = c
            action, di = design.assign_treatment(ds, state)
            new_obs, new_state, reward, _, info = env.step(k, state, action, params)
            xo = Observation(obs=new_obs, action=action, reward=reward, info=info, design_info=di)
            c2 = (new_obs, new_state, {"e": est.update(env, params, design, es["e"], xo)},
                  design.update(ds, xo))
            return c2, (obs, new_obs, action, reward, info["action_A"], info["action_B"])
        carry, log = jax.lax.scan(f, carry, jax.random.split(key, ch))
        st = est.end_chunk(env, params, design, carry[2]["e"])
        return (carry[0], carry[1], {"e": st}, carry[3]), log

    logs = []
    for c in range(nch):
        carry, log = run_chunk(carry, jax.random.PRNGKey(100 + c))
        logs.append(jax.tree.map(np.asarray, log))
    log = [np.concatenate([l[i] for l in logs]) for i in range(6)]
    st = carry[2]["e"]
    online = np.asarray(jax.jit(lambda s: est.estimate(env, params, design, s))(st))
    return env, params, est, st, log, online


@pytest.fixture(scope="module")
def mixed():
    est = PoolOPEEstimator(p=0.5, chunk=CH, thresholds=(0.0, 0.1), cv_blocks=2,
                           ridges=(1e-6, 1e-3, 1.0), gq_etas=(1e-9, 1e-6, 1e-3),
                           alphas=(0.01, 0.1), gq_eta=1e-3, gq_beta_ratio=2.0)
    return run(0.5, est)


def replay_features(env, params, est, log):
    pre, post, z, r, cA, cB = log
    prev = np.concatenate([np.zeros_like(post[:1]), post[:-1]]).astype(np.int32)

    def one(po, o, a, b):
        ob = SimpleNamespace(obs=o, info={"action_A": a, "action_B": b})
        phi, phic, rc, _, _ = est.step_features(params, po, ob)
        return phi, phic, rc
    Phi, Phic, rc = jax.jit(jax.vmap(one))(jnp.asarray(prev), jnp.asarray(post),
                                          jnp.asarray(cA), jnp.asarray(cB))
    return (np.asarray(Phi, np.float64), np.asarray(Phic, np.float64),
            np.asarray(rc, np.float64))


def numpy_targets(Phi, Phic, r, rc, z, d, p):
    """Independent (loop-free numpy) version of the per-step target samples."""
    out = []
    for pol in ("A", "B"):                       # cf
        cf = (z == 1) & d if pol == "A" else (z == 0) & d
        out.append((np.ones_like(r), np.where(cf[:, None], Phic, Phi), np.where(cf, rc, r)))
    for pol in ("A", "B"):                       # is
        w = np.where(d, (1 - z) / (1 - p) if pol == "A" else z / p, 1.0)
        out.append((w, Phi, r))
    return out


def test_accumulators_match_numpy(mixed):
    env, params, est, st, log, online = mixed
    pre, post, z, r, cA, cB = log
    z, r = z.astype(np.float64), r.astype(np.float64)
    Phi, Phic, rc = replay_features(env, params, est, log)
    n = len(r)
    d = (cA != cB)
    d[0] = False
    X0, X1 = Phi[:-1], Phi[1:]                    # transitions t = 1..n-1
    tg = numpy_targets(Phi, Phic, r, rc, z, d, est.p)
    blk = {k: np.asarray(v, np.float64) for k, v in st.blk.items()}
    A_on, b_on = blk["A"].sum(0), blk["b"].sum(0)
    for k, (c, Pn, rn) in enumerate(tg):
        c, Pn, rn = c[1:], Pn[1:], rn[1:]
        Dk = X0 - Pn
        Dk[:, -1] = 1.0
        A = (X0 * c[:, None]).T @ Dk
        b = (X0 * c[:, None]).T @ rn
        relA = np.abs(A - A_on[k]).max() / np.abs(A).max()
        relb = np.abs(b - b_on[k]).max() / np.abs(b).max()
        print("combo", k, "A rel err", relA, "b rel err", relb)
        assert relA < 1e-4 and relb < 1e-4
    C = X0.T @ X0
    assert np.abs(C - blk["C"].sum(0)).max() / np.abs(C).max() < 1e-4
    assert blk["n"].sum() == n - 1
    assert (blk["n"] > 0).sum() == 2              # interleaved CV blocks

    # ---- online TD / Diff-GQ1 replay (float64) ----
    F = Phi[:CH, :-1]
    mu, sd = F.mean(0), np.maximum(F.std(0), est.online_sd_floor)
    rmu, rsd = r[:CH].mean(), max(r[:CH].std(), 1e-3)
    assert np.allclose(mu, np.asarray(st.mu), rtol=1e-4, atol=1e-4)
    assert np.allclose(sd, np.asarray(st.sd), rtol=1e-4, atol=1e-4)
    D = Phi.shape[1] - 1
    al = np.asarray(est.alphas)
    G = len(al)
    th, R = np.zeros((4, G, D)), np.zeros((4, G))
    u, nu = np.zeros((4, G, D + 1)), np.zeros((4, G, D + 1))
    th_sum, u_sum, cnt = np.zeros_like(th), np.zeros((4, G, D)), 0
    for t in range(CH, n):                        # updates start after the first chunk
        x = (Phi[t - 1, :-1] - mu) / sd
        y = np.concatenate([[1.0], x])
        for k, (c, Pn, rn) in enumerate(tg):
            xn = (Pn[t, :-1] - mu) / sd
            rt = (rn[t] - rmu) / rsd
            for g in range(G):
                dl = rt - R[k, g] + th[k, g] @ (xn - x)
                step = al[g] / D * c[t] * dl
                th[k, g] = th[k, g] + step * x
                R[k, g] += est.td_eta * step
                dl2 = rt - u[k, g, 0] + u[k, g, 1:] @ (xn - x)
                ynu = nu[k, g] @ y
                a1 = al[g] / (D + 1)
                nu[k, g] = nu[k, g] + est.gq_beta_ratio * a1 * (c[t] * dl2 - ynu) * y
                grad = np.concatenate([[1.0], x - xn])
                reg = np.concatenate([[0.0], u[k, g, 1:]])
                u[k, g] = u[k, g] + a1 * (c[t] * ynu * grad - est.gq_eta * reg)
        th_sum += th
        u_sum += u[..., 1:]
        cnt += 1
    for name, ref, got in [("td_th", th, st.td_th), ("td_R", R, st.td_R), ("gq_u", u, st.gq_u),
                           ("gq_nu", nu, st.gq_nu), ("td_sum", th_sum, st.td_sum),
                           ("gq_sum", u_sum, st.gq_sum)]:
        got = np.asarray(got, np.float64)
        err = np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-8)
        print(name, "rel err", err)
        assert err < 2e-3, name
    assert float(st.n_avg) == cnt

    # ---- estimate: online pure_callback == numpy on the same stats; LSTD by hand ----
    ref = np_estimate(est.np_cfg, *[np.asarray(a) for a in est.host_args(st)])
    assert np.allclose(online, ref, rtol=1e-4, atol=1e-3, equal_nan=True)
    lab = {l: v for l, v in zip(est.labels, online)}
    mu_x, sd_x = X0.mean(0), X0.std(0)
    sd_x = np.where(sd_x > 1e-6, sd_x, 1.0)
    sd_x[-1] = 1.0
    I0 = np.eye(D + 1)
    I0[-1, -1] = 0
    for lam in est.ridges:
        rho = []
        for k in (0, 1):
            c, Pn, rn = [a[1:] for a in tg[k]]
            Dk = X0 - Pn
            Dk[:, -1] = 1.0
            As = (X0 * c[:, None]).T @ Dk / (n - 1) / np.outer(sd_x, sd_x)
            bs = (X0 * c[:, None]).T @ rn / (n - 1) / sd_x
            rho.append(np.linalg.solve(As + lam * I0, bs)[-1])
        print("lstd_cf", lam, rho[1] - rho[0], lab[f"lstd_cf_r{lam:g}"])
        assert abs((rho[1] - rho[0]) - lab[f"lstd_cf_r{lam:g}"]) < 1e-2 * max(1, abs(rho[1] - rho[0]))
    # large ridge -> theta ~ 0 -> direct method: mean(r^B - r^A) over transitions
    dm = np.mean(tg[1][2][1:] - tg[0][2][1:])
    big = PoolOPEEstimator(chunk=CH, ridges=(1e6,), gq_etas=(1e6,), alphas=(0.01,))
    R_ = np_rhos(big.np_cfg, *[np.asarray(a) for a in est.host_args(st)][:6],
                 np.zeros((4, 1, D)), np.zeros((4, 1)), np.zeros((4, 1, D)), np.zeros((4, 1)),
                 0.0, np.ones(D), 0.0, 1.0)
    assert abs((R_[("lstd", 1)][0][0] - R_[("lstd", 0)][0][0]) - dm) < 1e-3 * abs(dm)
    assert abs((R_[("td", 1)][0][0] - R_[("td", 0)][0][0]) - dm) < 1e-6 * abs(dm)
    assert np.isfinite(online[[i for i, l in enumerate(est.labels) if "_cv" not in l]]).all()


def test_on_policy_rho_equals_average_reward():
    """p = 1: every request gets arm B, so the data are on-policy for B. Every
    rho(B) estimate then equals the empirical average reward plus the
    telescoping term theta.(phi_T - phi_b)/n, which is small once the
    initial fill-up transient of the fleet is dropped (burn_in); cf and is
    coincide exactly."""
    ch = 2000
    BI = ch
    est = PoolOPEEstimator(p=1.0, chunk=ch, thresholds=(0.0, 0.1), cv_blocks=2, burn_in=BI,
                           ridges=(1e-6, 1e-4, 1e-2, 1.0), gq_etas=(1e-9, 1e-6, 1e-3),
                           alphas=(0.01, 0.1))
    env, params, est, st, log, online = run(1.0, est, nch=6, seed=1, ch=ch)
    r = log[3].astype(np.float64)
    rbar = r[BI:].mean()
    assert float(np.asarray(st.blk["n"]).sum()) == len(r) - BI
    R = np_rhos(est.np_cfg, *[np.asarray(a) for a in est.host_args(st)])
    for m in ("lstd", "gqfp", "td", "gq"):
        cf, is_ = np.asarray(R[(m, 1)][0]), np.asarray(R[(m, 3)][0])
        print(m, "rho_B", cf, "is", is_, "avg reward", rbar)
        assert np.allclose(cf, is_, rtol=1e-6, atol=1e-6)
        assert np.all(np.abs(cf - rbar) < 0.02 * abs(rbar)), m
    print("td iterate", R[("td", 1)][1], "gq iterate", R[("gq", 1)][1])
