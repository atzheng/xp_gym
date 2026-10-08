"""LSTD Differences-in-Q estimators for the pooled rideshare env.

U(y) = theta^T phi(y) is the post-decision value of the fleet, fitted by
LSTD on the experiment's own trajectory (mixed p policy): transitions
phi(y_{t-1}) -> phi(y_t) with reward r_t.  The estimates are all linear in
theta, so the estimator keeps O(D^2) LSTD statistics plus O(D) accumulators
and solves once per `estimate` call.

Variants (returned as a vector, see `labels`):
  paired     sum_t s_t d_t [ (r_t - r^cf_t) - gamma (U(y^cf_t) - U(y_t)) ] / T
             (needs the other arm's car, i.e. use_known_actions, to build y^cf_t)
  tr{lam}    sum_t delta_t e_t / T, with TD error
             delta_t = r_t - rbar + gamma U(y_t) - U(y_{t-1}) and trace
             e_t = lam*gamma e_{t-1} + w_t, w_t = (z_t/p - (1-z_t)/(1-p)) d_t.
             lam=0 is IPW LSTD-DQ with a value control variate; lam->1 is
             truncated-MC DQ with an LSTD baseline.
where s_t = +1 if the B arm was canonical else -1 and d_t = 1[arms chose
different cars].  The tr{lam} family is IPW-weighted generalized advantage
estimation (GAE): sum_j w_j sum_k (lam gamma)^k delta_{j+k}.

With jackknife_blocks = K > 0, the LSTD statistics are also kept per block of
consecutive chunks, and every variant gets a delete-one-block jackknife
bias-corrected version (suffix _jk):  K' tau(theta) - (K'-1) mean_k tau(theta_{-k})
over the K' non-empty blocks.  This targets LSTD's finite-sample bias.

To keep the GPU cost bandwidth-light, per-step feature rows are buffered and
the O(D^2) statistics are updated with one matmul per chunk in `end_chunk`
(called by the simulator after every `estimate_every_n_steps` steps, which
must equal `chunk`).
"""
from typing import Tuple
import numpy as np
from flax import struct
import jax
import jax.numpy as jnp

from or_gymnax.rideshare_pool import obs_to_state, insert_and_optimize_trip
from xp_gym.estimators.estimator import Estimator, EstimatorState
from xp_gym.estimators.pool_features import (
    pool_features, load_node_to_zone, reference_requests, match_costs,
    match_features_from_costs, feature_names, match_feature_names,
)

HI = jax.lax.Precision.HIGHEST


@struct.dataclass
class PoolLSTDDQState(EstimatorState):
    t: jnp.ndarray
    prev_obs: jnp.ndarray
    prev_phi: jnp.ndarray       # phi(y_{t-1}) incl. intercept
    chunk_first: jnp.ndarray    # phi(y) preceding the current chunk's first row
    buf: jnp.ndarray            # (chunk, D1) phi(y_t) rows of current chunk
    buf_r: jnp.ndarray          # (chunk,) r_t
    C: jnp.ndarray              # sum phi_{t-1} phi_t^T
    Cd: jnp.ndarray             # sum phi_{t-1} (phi_{t-1} - phi_t)^T
    b1: jnp.ndarray             # sum phi_{t-1} r_t
    s1: jnp.ndarray             # sum phi_{t-1}
    s2: jnp.ndarray             # sum phi_{t-1}^2
    sr: jnp.ndarray             # sum r_t (transitions)
    n: jnp.ndarray              # number of transitions
    P_r: jnp.ndarray            # paired: sum s d (r - r_cf)
    P_phi: jnp.ndarray          # paired: sum s d (phi_cf - phi)
    e: jnp.ndarray              # (L,) traces
    T_r: jnp.ndarray            # (L,) sum e r
    T_e: jnp.ndarray            # (L,) sum e
    T_phi: jnp.ndarray          # (L, D1) sum e (gamma phi_t - phi_{t-1})
    blk: dict                   # per-block copies of (C, Cd, b1, s1, s2, sr, n), leading axis K


@struct.dataclass
class PoolLSTDDQEstimator(Estimator):
    gamma: float = 1.0
    ridge: float = 1e-3
    p: float = 0.5
    trace_lambdas: Tuple[float, ...] = struct.field(pytree_node=False, default=(0.0, 0.9))
    chunk: int = struct.field(pytree_node=False, default=10000)
    # savings thresholds of arms (A, B), used by the match features
    thresholds: Tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 0.1))
    use_zones: bool = struct.field(pytree_node=False, default=True)   # ntrips + zone counts
    use_drop: bool = struct.field(pytree_node=False, default=False)   # dropoffs by zone x time
    use_busy: bool = struct.field(pytree_node=False, default=False)   # busy-time histograms
    match_K: int = struct.field(pytree_node=False, default=0)         # reference requests
    match_offsets: Tuple[int, ...] = struct.field(pytree_node=False, default=(0,))
    match_K_later: int = struct.field(pytree_node=False, default=64)  # refs at offsets > 0
    jackknife_blocks: int = struct.field(pytree_node=False, default=0)
    jackknife_block_chunks: int = struct.field(pytree_node=False, default=10)

    def __post_init__(self):
        # hydra passes ListConfigs; static fields must be hashable
        for k in ("trace_lambdas", "thresholds", "match_offsets"):
            object.__setattr__(self, k, tuple(float(x) if k != "match_offsets" else int(x)
                                              for x in getattr(self, k)))

    @property
    def labels(self):
        base = ["paired"] + [f"tr{l}" for l in self.trace_lambdas]
        return base + ([f"{b}_jk" for b in base] if self.jackknife_blocks else [])

    # -- features ---------------------------------------------------------
    def _setup(self, env_params):
        ip = env_params.env_params
        n2z, nz = load_node_to_zone()
        names = feature_names(nz)
        idx = []
        for i, nm in enumerate(names):
            if self.use_zones and (nm.startswith(("ntrips", "idle_", "one_final_", "full_drop_"))):
                idx.append(i)
            elif self.use_drop and nm.startswith("drop_"):
                idx.append(i)
            elif self.use_busy and nm.startswith(("busy", "firstfree")):
                idx.append(i)
        refs = reference_requests(ip.events, self.match_K) if self.match_K else (None, None)
        return ip, n2z, nz, jnp.asarray(idx, dtype=jnp.int32), refs

    def _ks(self):
        return [self.match_K if o == 0 else min(self.match_K, self.match_K_later)
                for o in self.match_offsets]

    def n_features(self, env_params):
        _, _, _, idx, _ = self._setup(env_params)
        nm = sum(4 * k for k in self._ks()) if self.match_K else 0
        return len(idx) + nm + 1

    def _features(self, setup, W, T, tau, changed=None, base=None):
        ip, n2z, nz, idx, (rs, rd) = setup
        f = [pool_features(W, T, tau, n2z, nz)[idx]]
        costs_out = []
        if self.match_K:
            for i, (o, k) in enumerate(zip(self.match_offsets, self._ks())):
                if base is None:
                    c = match_costs(W, T, tau + o, ip.distances, rs[:k], rd[:k], ip.max_active_trips)
                else:
                    sub = match_costs(W[changed], T[changed], tau + o, ip.distances, rs[:k], rd[:k],
                                      ip.max_active_trips)
                    c = tuple(b.at[:, changed].set(x) for b, x in zip(base[i], sub))
                costs_out.append(c)
                f.append(match_features_from_costs(*c, ip.distances, rs[:k], rd[:k],
                                                   tuple(self.thresholds), ip.profit_margin))
        f.append(jnp.ones(1))
        return jnp.concatenate(f).astype(jnp.float32), costs_out

    # -- Estimator API ----------------------------------------------------
    def reset(self, rng, env, env_params, design):
        D1 = self.n_features(env_params)
        L = len(self.trace_lambdas)
        obs_dim = 3 + 2 * env_params.env_params.n_cars * 2 * env_params.env_params.max_active_trips
        z = lambda *s: jnp.zeros(s, jnp.float32)
        K = max(self.jackknife_blocks, 1)
        blk = dict(C=z(K, D1, D1), Cd=z(K, D1, D1), b1=z(K, D1), s1=z(K, D1), s2=z(K, D1),
                   sr=z(K), n=z(K))
        return PoolLSTDDQState(blk=blk,
            t=jnp.array(0), prev_obs=jnp.zeros(obs_dim, jnp.int32),
            prev_phi=z(D1), chunk_first=z(D1), buf=z(self.chunk, D1), buf_r=z(self.chunk),
            C=z(D1, D1), Cd=z(D1, D1), b1=z(D1), s1=z(D1), s2=z(D1), sr=z(), n=z(),
            P_r=z(), P_phi=z(D1), e=z(L), T_r=z(L), T_e=z(L), T_phi=z(L, D1))

    def update(self, env, env_params, design, state, obs):
        setup = self._setup(env_params)
        ip = setup[0]
        n_cars, nwp = ip.n_cars, 2 * ip.max_active_trips
        ev_next, W, T = obs_to_state(n_cars, nwp, obs.obs)
        tau = ev_next.t
        phi, costs = self._features(setup, W, T, tau)

        # counterfactual post-decision fleet (other arm's car instead)
        ev, Wp, Tp = obs_to_state(n_cars, nwp, state.prev_obs)
        cA = obs.info["action_A"].reshape(-1)[0]
        cB = obs.info["action_B"].reshape(-1)[0]
        cb = jnp.maximum(cB, 0)
        wp_cf, t_cf, mc_cf, _ = insert_and_optimize_trip(
            ip.distances, Wp[cb], Tp[cb], ev.src, ev.dest, ev.t, ip.max_active_trips)
        found = cB >= 0
        Wc = jnp.where(found, Wp.at[cb].set(wp_cf), Wp)
        Tc = jnp.where(found, Tp.at[cb].set(t_cf), Tp)
        changed = jnp.stack([jnp.maximum(cA, 0), cb])
        phi_cf, _ = self._features(setup, Wc, Tc, tau, changed, costs if self.match_K else None)
        r = obs.reward.astype(jnp.float32)
        r_cf = jnp.where(found, ip.distances[ev.src, ev.dest] * (1 + ip.profit_margin) - mc_cf, 0.0)

        z = obs.action.astype(jnp.float32)
        s = 2 * z - 1
        has_prev = state.t > 0
        d = ((cA != cB) & has_prev).astype(jnp.float32)
        w = (z / self.p - (1 - z) / (1 - self.p)) * d

        lam = jnp.asarray(self.trace_lambdas, jnp.float32)
        e = lam * self.gamma * state.e + w
        dphi_td = self.gamma * phi - state.prev_phi
        hp = has_prev.astype(jnp.float32)

        i = state.t % self.chunk
        return state.replace(
            t=state.t + 1,
            prev_obs=obs.obs.astype(jnp.int32),
            prev_phi=phi,
            buf=state.buf.at[i].set(phi),
            buf_r=state.buf_r.at[i].set(r),
            P_r=state.P_r + s * d * (r - r_cf),
            P_phi=state.P_phi + s * d * (phi_cf - phi),
            e=e,
            T_r=state.T_r + hp * e * r,
            T_e=state.T_e + hp * e,
            T_phi=state.T_phi + hp * e[:, None] * dphi_td[None, :],
        )

    def end_chunk(self, env, env_params, design, state):
        X1 = state.buf
        X0 = jnp.concatenate([state.chunk_first[None], X1[:-1]], 0)
        # the very first row of the run has no predecessor
        first = (state.t == self.chunk)
        m = jnp.ones(self.chunk).at[0].set(jnp.where(first, 0.0, 1.0))
        X0m = X0 * m[:, None]
        dot = lambda a, b: jnp.dot(a, b, precision=HI)
        inc = dict(C=dot(X0m.T, X1), Cd=dot(X0m.T, X0 - X1), b1=dot(X0m.T, state.buf_r),
                   s1=X0m.sum(0), s2=(X0m ** 2).sum(0), sr=(m * state.buf_r).sum(), n=m.sum())
        new = {k: getattr(state, k) + v for k, v in inc.items()}
        blk = state.blk
        if self.jackknife_blocks:
            k = jnp.minimum((state.t // self.chunk - 1) // self.jackknife_block_chunks,
                            self.jackknife_blocks - 1)
            blk = {key: blk[key].at[k].add(inc[key]) for key in blk}
        return state.replace(chunk_first=X1[-1], blk=blk, **new)

    def theta(self, state, leave_out=None):
        """Ridge-regularised LSTD solution on standardised features. The
        system is badly conditioned at gamma=1, so it is solved in float64
        on the host."""
        def solve(Cd, C, b1, s1, s2, sr, n, gamma, ridge):
            Cd, C, b1, s1, s2 = (np.asarray(x, np.float64) for x in (Cd, C, b1, s1, s2))
            n = max(float(n), 1.0)
            rbar = float(sr) / n
            sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
            sd = np.where(sd > 1e-6, sd, 1.0); sd[-1] = 1.0
            A = (Cd + (1 - float(gamma)) * C) / n / np.outer(sd, sd)
            b = (b1 - s1 * rbar) / n / sd
            th = np.linalg.solve(A + float(ridge) * np.eye(len(A)), b) / sd
            return th.astype(np.float32), np.float32(rbar)

        def solve_batched(*args):
            # vmap_method="expand_dims": leading axis may be a batch axis
            Cd = np.asarray(args[0])
            if Cd.ndim == 2:
                return solve(*args)
            B = Cd.shape[0]
            bc = lambda x, i: np.broadcast_to(np.asarray(x), (B,) + np.asarray(x).shape[1:])[i]
            outs = [solve(*(bc(a, i) for a in args)) for i in range(B)]
            return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])

        D1 = state.Cd.shape[-1]
        shapes = (jax.ShapeDtypeStruct((D1,), jnp.float32), jax.ShapeDtypeStruct((), jnp.float32))
        stats = (state.Cd, state.C, state.b1, state.s1, state.s2, state.sr, state.n)
        if leave_out is not None:
            b = state.blk
            stats = tuple(x - b[k][leave_out] for x, k in
                          zip(stats, ["Cd", "C", "b1", "s1", "s2", "sr", "n"]))
        return jax.pure_callback(
            solve_batched, shapes, *stats, jnp.float32(self.gamma), jnp.float32(self.ridge),
            vmap_method="expand_dims")

    def _taus(self, state, th, rbar):
        T = jnp.maximum(state.t, 1).astype(jnp.float32)
        paired = (state.P_r - self.gamma * jnp.dot(th, state.P_phi, precision=HI)) / T
        tr = (state.T_r - rbar * state.T_e + jnp.dot(state.T_phi, th, precision=HI)) / T
        return jnp.concatenate([paired[None], tr])

    def estimate(self, env, env_params, design, state):
        th, rbar = self.theta(state)
        tau = self._taus(state, th, rbar)
        if not self.jackknife_blocks:
            return tau
        # leave-one-block-out; rbar kept at the full-data value
        loo = jnp.stack([self._taus(state, self.theta(state, k)[0], rbar)
                         for k in range(self.jackknife_blocks)])
        nonempty = state.blk["n"] > 0
        Ke = jnp.maximum(nonempty.sum(), 1).astype(jnp.float32)
        mean_loo = jnp.sum(jnp.where(nonempty[:, None], loo, 0.0), 0) / Ke
        tau_jk = jnp.where(Ke > 1, Ke * tau - (Ke - 1) * mean_loo, tau)
        return jnp.concatenate([tau, tau_jk])
