"""Off-policy evaluation (OPE) of the all-A and all-B policies' average reward
from a unit-randomized experiment on the pooled rideshare env.

    ATE_hat = rho_hat(B) - rho_hat(A),

where rho_hat(pi) is an off-policy estimate of pi's long-run average reward per
request, with a linear post-decision fleet value U_pi(y) = theta_pi^T phi(y)
(the "zones" features of lstd_dq_pool / pool_features).

Post-decision MDP.  y_{t-1} is the fleet after request t-1 was dispatched; the
next request xi_t is exogenous (its draw does not depend on the action), the
policy picks a car, giving r_t and y_t.  The average-reward Bellman equation
    U(y) + rho = E[ r^pi(y, xi) + U(y^pi(y, xi)) ]
holds pointwise, so for ANY sampling distribution of y (here the behaviour's)
    E_b[ x (r^pi - rho + U(y'^pi) - U(y)) ] = 0,   x = [phi(y), 1].     (*)
With phi(y) carrying an intercept, (*) is a (D+1)-dim linear system in
w = (theta without intercept, rho): the off-policy average-reward TD fixed
point (Zhang, Wan, Sutton & Whiteson 2021, "Average-Reward Off-Policy Policy
Evaluation with Function Approximation", Sec. 3; Yu & Bertsekas 2009).

Two sample formulations of the per-step target transition (y_{t-1} -> y'^pi):
  cf  "counterfactual / known action" (recommended).  The env reports both
      arms' cars (info action_A = dispatched car, action_B = the other arm's
      car) and the dispatch is deterministic given the car, so the target-pi
      transition from y_{t-1} is known at every step: r^pi_t, y^pi_t are the
      observed ones if pi's car was dispatched, otherwise the counterfactual
      ones (other car inserted into the pre-decision fleet, exactly as in
      lstd_dq_pool's `paired`).  Weight c_t = 1.
  is  per-step importance sampling of the observed transition, with
      c^A_t = 1 if the arms agree else (1-z_t)/(1-p),
      c^B_t = 1 if the arms agree else z_t/p.
  Since xi_t is exogenous and pi acts only at step t within a transition,
  E_z[c_t g(observed)] = g(target transition): the cf sample is the
  conditional expectation (Rao-Blackwellisation) of the is sample.  Same fixed
  point, smaller variance, no zero-weight steps.  Neither corrects the
  behaviour-vs-target state-distribution shift; that only matters through the
  linear approximation error (with an exact U, (*) holds under any d_b).
  NB: with ONE shared theta, cf's rho(B) - rho(A) is exactly lstd_dq_pool's
  `paired` DQ estimate; here each policy gets its own theta.

Outputs (ATE columns, see `labels`; f in {cf, is}):
  lstd_{f}_r{lam}   ridge LSTD: (A + lam I0) w = b on standardized features,
                    A = E[c x (x - phi'^pi + e_rho)^T], b = E[c x r^pi],
                    I0 = identity without the rho entry (rho is not shrunk)
  lstd_{f}_cv       lam picked per policy by K-fold block CV of the held-out
                    MSPBE ||b_j - A_j w_{-j}||^2_{C^+}, blocks = chunks
                    interleaved mod K; + _rhoA / _rhoB (the two rho's)
  gqfp_{f}_e{eta}   Diff-GQ1 fixed point (Zhang et al. 2021, objective behind
                    eq. (11)): argmin_w ||b - A w||^2_{C^+} + eta ||theta||^2,
                    C = E[x x^T]  ->  (A^T C^+ A + eta I0) w = A^T C^+ b
  gqfp_{f}_cv       eta picked by the same block CV
  td_{f}_a{alpha}   online off-policy differential semi-gradient TD(0) (Wan,
                    Naik & Sutton 2021, Differential TD with IS ratio c):
                      delta = r~ - Rbar + theta.(x~' - x~)
                      theta += (alpha/D) c delta x~,  Rbar += td_eta (alpha/D) c delta
                    Column = Bellman plug-in rho = sum c[r^pi + U(y') - U(y)] / sum c
                    (the last row of (*)) at the Polyak-averaged theta.
  tdit_{f}_a{alpha} the Rbar iterate (Wan et al.'s own estimate of rho).
  td_{f}_sel        alpha picked per policy by the full-data MSPBE of the
                    averaged iterate (in-sample heuristic).
  gq_{f}_a{alpha}, gqit_{f}_a{alpha}, gq_{f}_sel: online Diff-GQ1 (eq. (11))
                    with u = (rbar, w), y = (1, x~), normalized steps:
                      delta = r~ - rbar + w.(x~' - x~)
                      nu += (gq_beta_ratio alpha/D1) (c delta - y.nu) y
                      u  += (alpha/D1) [c (1, x~ - x~') (y.nu) - gq_eta (0, w)]
Online methods start after the first chunk with feature / reward means and
standard deviations frozen from that chunk (x~ = (phi - mu)/sd, r~ = (r - mr)/sr).
`burn_in` drops transitions t < burn_in everywhere (the fleet starts empty;
the fill-up transient otherwise enters rho through the telescoping term
theta.(phi_T - phi_0)/T of the rho equation).  Note demand is time-of-day
non-stationary, so individual rho's over short windows carry a similar
drift term; it is common to A and B and largely cancels in the ATE.

O(D^2) statistics are accumulated once per chunk via `end_chunk` (as in
lstd_dq_pool; requires chunk == estimate_every_n_steps); all solves run in
float64 on the host via pure_callback.
"""
from typing import Tuple
import numpy as np
from flax import struct
import jax
import jax.numpy as jnp

from or_gymnax.rideshare_pool import obs_to_state, insert_and_optimize_trip
from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.estimators.lstd_dq_pool import PoolLSTDDQEstimator

HI = jax.lax.Precision.HIGHEST
FORMS = ("cf", "is")
NCOMBO = 4  # (cf,A), (cf,B), (is,A), (is,B)


@struct.dataclass
class PoolOPEState(EstimatorState):
    t: jnp.ndarray
    prev_obs: jnp.ndarray
    prev_phi: jnp.ndarray
    chunk_first: jnp.ndarray
    buf: jnp.ndarray        # (chunk, D1) phi(y_t)
    bufc: jnp.ndarray       # (chunk, D1) phi(y^cf_t)
    buf_r: jnp.ndarray      # (chunk,)
    buf_rc: jnp.ndarray     # (chunk,) r^cf
    buf_z: jnp.ndarray      # (chunk,)
    buf_d: jnp.ndarray      # (chunk,) arms differ
    blk: dict               # per-block A (K,4,D1,D1), b (K,4,D1), C, s1, s2, n
    started: jnp.ndarray    # online methods running
    mu: jnp.ndarray         # (D,) frozen online standardization
    sd: jnp.ndarray
    rmu: jnp.ndarray
    rsd: jnp.ndarray
    td_th: jnp.ndarray      # (4, G, D)
    td_R: jnp.ndarray       # (4, G)
    td_sum: jnp.ndarray     # (4, G, D) sum of iterates (Polyak)
    gq_u: jnp.ndarray       # (4, G, D+1)
    gq_nu: jnp.ndarray      # (4, G, D+1)
    gq_sum: jnp.ndarray     # (4, G, D)
    n_avg: jnp.ndarray


def _fmt(x):
    return f"{x:g}"


def targets(phi, phic, r, rc, z, d, p):
    """Per-combo (weight c, next features phi', reward r') for the 4 combos
    (cf,A), (cf,B), (is,A), (is,B).  Works on single rows or (n,) batches.
    z = 1 iff arm B was dispatched (canonical); d = 1 iff the arms differ."""
    zd = z * d                 # B dispatched, A differs -> A is counterfactual
    nzd = (1 - z) * d          # A dispatched, B differs -> B is counterfactual
    phiA = zd[..., None] * phic + (1 - zd[..., None]) * phi
    phiB = nzd[..., None] * phic + (1 - nzd[..., None]) * phi
    rA = zd * rc + (1 - zd) * r
    rB = nzd * rc + (1 - nzd) * r
    inv1 = 1.0 / (1 - p) if p < 1 else 0.0
    inv0 = 1.0 / p if p > 0 else 0.0
    cA = jnp.where(d > 0, (1 - z) * inv1, 1.0)
    cB = jnp.where(d > 0, z * inv0, 1.0)
    one = jnp.ones_like(r)
    c = jnp.stack([one, one, cA * one, cB * one], 0)
    ph = jnp.stack([phiA, phiB, phi, phi], 0)
    rr = jnp.stack([rA, rB, r, r], 0)
    return c, ph, rr


@struct.dataclass
class PoolOPEEstimator(Estimator):
    p: float = struct.field(pytree_node=False, default=0.5)
    chunk: int = struct.field(pytree_node=False, default=10000)
    thresholds: Tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 0.1))
    use_zones: bool = struct.field(pytree_node=False, default=True)
    use_drop: bool = struct.field(pytree_node=False, default=False)
    use_busy: bool = struct.field(pytree_node=False, default=False)
    ridges: Tuple[float, ...] = struct.field(
        pytree_node=False, default=tuple(10.0 ** k for k in range(-8, 2)))
    gq_etas: Tuple[float, ...] = struct.field(
        pytree_node=False, default=tuple(10.0 ** k for k in range(-12, 0)))
    alphas: Tuple[float, ...] = struct.field(
        pytree_node=False, default=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0))
    td_eta: float = struct.field(pytree_node=False, default=1.0)      # Rbar step ratio
    gq_eta: float = struct.field(pytree_node=False, default=0.0)      # online Diff-GQ1 ridge
    gq_beta_ratio: float = struct.field(pytree_node=False, default=1.0)
    online_sd_floor: float = struct.field(pytree_node=False, default=1.0)
    cv_blocks: int = struct.field(pytree_node=False, default=5)
    burn_in: int = struct.field(pytree_node=False, default=0)  # drop transitions t < burn_in

    def __post_init__(self):
        for k in ("ridges", "gq_etas", "alphas", "thresholds"):
            object.__setattr__(self, k, tuple(float(x) for x in getattr(self, k)))

    @property
    def labels(self):
        out = []
        for f in FORMS:
            out += [f"lstd_{f}_r{_fmt(l)}" for l in self.ridges]
            out += [f"lstd_{f}_cv", f"lstd_{f}_cv_rhoA", f"lstd_{f}_cv_rhoB"]
        for f in FORMS:
            out += [f"gqfp_{f}_e{_fmt(e)}" for e in self.gq_etas] + [f"gqfp_{f}_cv"]
        for m in ("td", "gq"):
            for f in FORMS:
                out += [f"{m}_{f}_a{_fmt(a)}" for a in self.alphas]
                out += [f"{m}it_{f}_a{_fmt(a)}" for a in self.alphas]
                out += [f"{m}_{f}_sel"]
        return out

    # -- features (reuse lstd_dq_pool's) -----------------------------------
    def _fe(self):
        return PoolLSTDDQEstimator(thresholds=self.thresholds, use_zones=self.use_zones,
                                   use_drop=self.use_drop, use_busy=self.use_busy,
                                   chunk=self.chunk)

    def reset(self, rng, env, env_params, design):
        D1 = self._fe().n_features(env_params)
        D = D1 - 1
        G = len(self.alphas)
        K = max(self.cv_blocks, 1)
        ip = env_params.env_params
        obs_dim = 3 + 2 * ip.n_cars * 2 * ip.max_active_trips
        z = lambda *s: jnp.zeros(s, jnp.float32)
        blk = dict(A=z(K, NCOMBO, D1, D1), b=z(K, NCOMBO, D1), C=z(K, D1, D1),
                   s1=z(K, D1), s2=z(K, D1), n=z(K))
        return PoolOPEState(
            t=jnp.array(0), prev_obs=jnp.zeros(obs_dim, jnp.int32), prev_phi=z(D1),
            chunk_first=z(D1), buf=z(self.chunk, D1), bufc=z(self.chunk, D1),
            buf_r=z(self.chunk), buf_rc=z(self.chunk), buf_z=z(self.chunk), buf_d=z(self.chunk),
            blk=blk, started=jnp.array(False), mu=z(D), sd=jnp.ones(D, jnp.float32),
            rmu=z(), rsd=jnp.ones((), jnp.float32),
            td_th=z(NCOMBO, G, D), td_R=z(NCOMBO, G), td_sum=z(NCOMBO, G, D),
            gq_u=z(NCOMBO, G, D + 1), gq_nu=z(NCOMBO, G, D + 1), gq_sum=z(NCOMBO, G, D),
            n_avg=z())

    def step_features(self, env_params, prev_obs, obs):
        """phi(y_t), phi(y^cf_t), r^cf_t and the env's (dispatched, other) cars."""
        fe = self._fe()
        setup = fe._setup(env_params)
        ip = setup[0]
        n_cars, nwp = ip.n_cars, 2 * ip.max_active_trips
        ev_next, W, T = obs_to_state(n_cars, nwp, obs.obs)
        tau = ev_next.t
        phi, _ = fe._features(setup, W, T, tau)
        ev, Wp, Tp = obs_to_state(n_cars, nwp, prev_obs)
        cA = obs.info["action_A"].reshape(-1)[0]   # dispatched (canonical) car
        cB = obs.info["action_B"].reshape(-1)[0]   # other arm's car
        cb = jnp.maximum(cB, 0)
        wp_cf, t_cf, mc_cf, _ = insert_and_optimize_trip(
            ip.distances, Wp[cb], Tp[cb], ev.src, ev.dest, ev.t, ip.max_active_trips)
        found = cB >= 0
        Wc = jnp.where(found, Wp.at[cb].set(wp_cf), Wp)
        Tc = jnp.where(found, Tp.at[cb].set(t_cf), Tp)
        phic, _ = fe._features(setup, Wc, Tc, tau)
        rc = jnp.where(found, ip.distances[ev.src, ev.dest] * (1 + ip.profit_margin) - mc_cf,
                       0.0).astype(jnp.float32)
        return phi, phic, rc, cA, cB

    def update(self, env, env_params, design, state, obs):
        phi, phic, rc, cA, cB = self.step_features(env_params, state.prev_obs, obs)
        r = obs.reward.astype(jnp.float32).reshape(())
        z = obs.action.astype(jnp.float32).reshape(())
        has_prev = state.t > 0
        d = ((cA != cB) & has_prev).astype(jnp.float32)

        # ---- online TD / Diff-GQ1 (after the first chunk) ----
        go = (state.started & has_prev & (state.t >= self.burn_in)).astype(jnp.float32)
        c, ph, rr = targets(phi, phic, r, rc, z, d, self.p)              # (4,), (4,D1), (4,)
        x = (state.prev_phi[:-1] - state.mu) / state.sd                  # (D,)
        xn = (ph[:, :-1] - state.mu) / state.sd                          # (4, D)
        rt = (rr - state.rmu) / state.rsd                                # (4,)
        D = x.shape[0]
        al = jnp.asarray(self.alphas, jnp.float32)[None, :]              # (1, G)
        dx = xn - x[None, :]                                             # (4, D)
        # differential TD
        th = state.td_th
        dlt = rt[:, None] - state.td_R + jnp.einsum("kgd,kd->kg", th, dx, precision=HI)
        g = go * c[:, None] * dlt * al / D                               # (4, G)
        th = th + g[..., None] * x[None, None, :]
        R = state.td_R + self.td_eta * g
        # Diff-GQ1
        u, nu = state.gq_u, state.gq_nu
        y = jnp.concatenate([jnp.ones(1), x])
        dlt2 = rt[:, None] - u[..., 0] + jnp.einsum("kgd,kd->kg", u[..., 1:], dx, precision=HI)
        ynu = jnp.einsum("kgd,d->kg", nu, y, precision=HI)
        a1 = go * al / (D + 1)
        nu = nu + (self.gq_beta_ratio * a1 * (c[:, None] * dlt2 - ynu))[..., None] * y
        grad = jnp.concatenate([jnp.ones((NCOMBO, 1)), -dx], 1)          # (1, x - x')
        reg = jnp.concatenate([jnp.zeros_like(u[..., :1]), u[..., 1:]], -1)
        u = u + a1[..., None] * ((c[:, None] * ynu)[..., None] * grad[:, None, :]
                                 - self.gq_eta * reg)

        i = state.t % self.chunk
        return state.replace(
            t=state.t + 1, prev_obs=obs.obs.astype(jnp.int32), prev_phi=phi,
            buf=state.buf.at[i].set(phi), bufc=state.bufc.at[i].set(phic),
            buf_r=state.buf_r.at[i].set(r), buf_rc=state.buf_rc.at[i].set(rc),
            buf_z=state.buf_z.at[i].set(z), buf_d=state.buf_d.at[i].set(d),
            td_th=th, td_R=R, td_sum=state.td_sum + go * th,
            gq_u=u, gq_nu=nu, gq_sum=state.gq_sum + go * u[..., 1:],
            n_avg=state.n_avg + go)

    def end_chunk(self, env, env_params, design, state):
        X1 = state.buf
        X0 = jnp.concatenate([state.chunk_first[None], X1[:-1]], 0)
        # global step index of each row; the very first row has no predecessor
        idx = state.t - self.chunk + jnp.arange(self.chunk)
        m = (idx >= max(self.burn_in, 1)).astype(jnp.float32)
        X0m = X0 * m[:, None]
        dot = lambda a, b: jnp.dot(a, b, precision=HI)
        c, ph, rr = targets(X1, state.bufc, state.buf_r, state.buf_rc, state.buf_z,
                            state.buf_d, self.p)                       # (4,n), (4,n,D1), (4,n)
        A, b = [], []
        for k in range(NCOMBO):
            Xw = X0m * c[k][:, None]
            Dk = (X0 - ph[k]).at[:, -1].set(1.0)    # rho column: x[-1] - phi'[-1] = 0 -> 1
            A.append(dot(Xw.T, Dk))
            b.append(dot(Xw.T, rr[k]))
        inc = dict(A=jnp.stack(A), b=jnp.stack(b), C=dot(X0m.T, X0), s1=X0m.sum(0),
                   s2=(X0m ** 2).sum(0), n=m.sum())
        kb = (state.t // self.chunk - 1) % max(self.cv_blocks, 1)
        blk = {key: state.blk[key].at[kb].add(inc[key]) for key in state.blk}
        # freeze the online standardization at the end of the first chunk with
        # at least half its rows past burn-in (masked mean / sd)
        mb = (idx >= self.burn_in).astype(jnp.float32)
        nb = jnp.maximum(mb.sum(), 1.0)
        F = X1[:, :-1]
        mu = (mb[:, None] * F).sum(0) / nb
        sd = jnp.sqrt((mb[:, None] * (F - mu) ** 2).sum(0) / nb)
        sd = jnp.maximum(sd, self.online_sd_floor)
        rmu = (mb * state.buf_r).sum() / nb
        rsd = jnp.maximum(jnp.sqrt((mb * (state.buf_r - rmu) ** 2).sum() / nb), 1e-3)
        keep = state.started | (mb.sum() < self.chunk / 2)
        return state.replace(
            chunk_first=X1[-1], blk=blk, started=~keep | state.started,
            mu=jnp.where(keep, state.mu, mu), sd=jnp.where(keep, state.sd, sd),
            rmu=jnp.where(keep, state.rmu, rmu), rsd=jnp.where(keep, state.rsd, rsd))

    # -- estimation (host, float64) ----------------------------------------
    def host_args(self, state):
        return (state.blk["A"], state.blk["b"], state.blk["C"], state.blk["s1"],
                state.blk["s2"], state.blk["n"], state.td_sum, state.td_R, state.gq_sum,
                state.gq_u[..., 0], state.n_avg, state.sd, state.rmu, state.rsd)

    @property
    def np_cfg(self):
        return dict(ridges=self.ridges, etas=self.gq_etas, G=len(self.alphas))

    def estimate(self, env, env_params, design, state):
        cfg = self.np_cfg
        L = len(self.labels)

        def host(*a):
            a = [np.asarray(x) for x in a]
            if a[5].ndim == 1:
                return np_estimate(cfg, *a).astype(np.float32)
            B = max(x.shape[0] for x in a)
            bc = lambda x, i: np.broadcast_to(x, (B,) + x.shape[1:])[i]
            return np.stack([np_estimate(cfg, *(bc(x, i) for x in a)) for i in range(B)]
                            ).astype(np.float32)

        return jax.pure_callback(host, jax.ShapeDtypeStruct((L,), jnp.float32),
                                 *self.host_args(state), vmap_method="expand_dims")


def np_rhos(cfg, A, b, C, s1, s2, n, td_sum, td_R, gq_sum, gq_r, n_avg, osd, rmu, rsd):
    """Per-combo rho estimates (combo k: 0 (cf,A), 1 (cf,B), 2 (is,A), 3 (is,B)):
    R[("lstd"|"gqfp", k)] = (rho over the grid, CV-selected rho);
    R[("td"|"gq", k)] = (plug-in rho over alphas, iterate rho, MSPBE-selected)."""
    f64 = lambda x: np.asarray(x, np.float64)
    A, b, C, s1, s2, n = map(f64, (A, b, C, s1, s2, n))
    Af, bf, Cf, nf = A.sum(0), b.sum(0), C.sum(0), float(n.sum())
    N = max(nf, 1.0)
    mu = s1.sum(0) / N
    sd = np.sqrt(np.maximum(s2.sum(0) / N - mu ** 2, 0.0))
    sd = np.where(sd > 1e-6, sd, 1.0)
    sd[-1] = 1.0
    D1 = len(sd)
    out_sd = np.outer(sd, sd)
    I0 = np.eye(D1)
    I0[-1, -1] = 0.0
    # MSPBE weight in standardized instrument coords, W = T^T Cc^+ T with
    # centered instruments (T) for conditioning (the objective is invariant to T)
    T = np.eye(D1)
    T[:-1, -1] = -(mu / sd)[:-1]
    Cs = Cf / N / out_sd
    W = T.T @ np.linalg.pinv(T @ Cs @ T.T, rcond=1e-10, hermitian=True) @ T
    W = (W + W.T) / 2

    def stdz(Ak, bk, nk):
        nk = max(nk, 1.0)
        return Ak / nk / out_sd, bk / nk / sd

    def lstd_prep(As, bs):
        return lambda lam: np.linalg.solve(As + lam * I0, bs)

    def gq_prep(As, bs):
        M, v = As.T @ W @ As, As.T @ W @ bs
        return lambda eta: np.linalg.solve(M + eta * I0, v)

    def safe(fn, lam):
        try:
            w = fn(lam)
            return w if np.all(np.isfinite(w)) else None
        except np.linalg.LinAlgError:
            return None

    def mspbe(As, bs, w):
        g = bs - As @ w
        return float(g @ W @ g)

    blocks = [j for j in range(len(n)) if n[j] > 0]

    def path_and_cv(k, prep, grid):
        fn = prep(*stdz(Af[k], bf[k], nf))
        ws = [safe(fn, lam) for lam in grid]
        rhos = np.array([w[-1] if w is not None else np.nan for w in ws])
        if len(blocks) < 2:
            return rhos, np.nan
        score = np.zeros(len(grid))
        for j in blocks:
            fn = prep(*stdz(Af[k] - A[j, k], bf[k] - b[j, k], nf - n[j]))
            Ate, bte = stdz(A[j, k], b[j, k], n[j])
            for gi, lam in enumerate(grid):
                w = safe(fn, lam)
                score[gi] += n[j] * mspbe(Ate, bte, w) if w is not None else np.inf
        score = np.where(np.isfinite(score), score, np.inf)
        return rhos, rhos[int(np.argmin(score))]

    R = {}
    for k in range(NCOMBO):
        R[("lstd", k)] = path_and_cv(k, lstd_prep, cfg["ridges"])
        R[("gqfp", k)] = path_and_cv(k, gq_prep, cfg["etas"])
    # online methods: plug-in rho at the averaged weights, iterate, MSPBE score
    G = cfg["G"]
    na = max(float(n_avg), 1.0)
    osd = f64(osd)
    for m, sums, its in (("td", f64(td_sum), f64(td_R)), ("gq", f64(gq_sum), f64(gq_r))):
        for k in range(NCOMBO):
            As, bs = stdz(Af[k], bf[k], nf)
            den = Af[k][-1, -1]
            plug, it, sc = np.zeros(G), np.zeros(G), np.zeros(G)
            for gi in range(G):
                th = float(rsd) * sums[k, gi] / na / osd           # raw coords (D,)
                rho = (bf[k][-1] - Af[k][-1, :-1] @ th) / den if den > 0 else np.nan
                w = np.concatenate([th * sd[:-1], [rho]])           # standardized coords
                plug[gi] = rho
                it[gi] = float(rmu) + float(rsd) * its[k, gi]
                s = mspbe(As, bs, w)
                sc[gi] = s if np.isfinite(s) else np.inf
            R[(m, k)] = (plug, it, plug[int(np.argmin(sc))])
    return R


def np_estimate(cfg, *args):
    """Pure-numpy estimate for one env from the accumulated statistics;
    returns the vector matching PoolOPEEstimator.labels (rho(B) - rho(A))."""
    R = np_rhos(cfg, *args)
    out = []
    for f in range(2):
        (rA, cvA), (rB, cvB) = R[("lstd", 2 * f)], R[("lstd", 2 * f + 1)]
        out += list(rB - rA) + [cvB - cvA, cvA, cvB]
    for f in range(2):
        (rA, cvA), (rB, cvB) = R[("gqfp", 2 * f)], R[("gqfp", 2 * f + 1)]
        out += list(rB - rA) + [cvB - cvA]
    for m in ("td", "gq"):
        for f in range(2):
            a, b = R[(m, 2 * f)], R[(m, 2 * f + 1)]
            out += list(b[0] - a[0]) + list(b[1] - a[1]) + [b[2] - a[2]]
    return np.asarray(out, np.float64)
