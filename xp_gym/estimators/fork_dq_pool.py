"""Fork DQ for the pooled rideshare env: n-step paired DQ with model-based
counterfactual rollouts and an LSTD tail.

At every step t where the two arms would dispatch differently, a "fork" is
opened: the counterfactual fleet in which the *other* arm acted at t.  The fork
is stored sparsely, as ghost copies of the cars whose state differs from the
real fleet (at most `fork_ghosts`).  At every later step the fork is replayed
on the realised request with the realised arm z_t' (same randomness as the real
world), using the known greedy dispatch rule, so the per-step reward difference
between the two worlds is exact.  At age H the remaining effect is bootstrapped
with the LSTD post-decision value U = theta^T phi (fitted exactly as in
PoolLSTDDQEstimator):

    Delta_t(H) = sum_{k<H} (r_{t+k} - r^fork_{t+k}) + U(y_{t+H-1}) - U(y^fork_{t+H-1})
    fork_H     = sum_t s_t Delta_t(H) / T,          s_t = +1 if arm B was real at t

H=1 is the paired LSTD-DQ estimator.  A fork that would need more than
`fork_ghosts` ghosts is closed early (bootstrapped at that step); when all
`fork_slots` are busy a new differing step falls back to H=1.  Ghosts whose
state becomes functionally identical to the real car (both idle at the same
node) are dropped.  The value features are per-car additive, so the bootstrap
needs only the ghost cars and their real counterparts.

Fork-TD (`fork_td=True`): the tail value is instead fitted on the forks
themselves.  With psi_a = phi(fork) - phi(real) at fork age a and
dr_a = r_real - r_fork at that step, the value difference
V(real) - V(fork) = -theta_f^T psi solves the one-step TD relation
-theta_f^T psi_a = dr_a - theta_f^T psi_{a+1}; LSTD on all fork transitions
gives  sum psi_a (psi_a - psi_{a+1})^T theta_f = -sum psi_a dr_a.  Both worlds
see the same requests, so the demand cycles that confound the on-trajectory
LSTD fit cancel, so time-of-day interactions can be used: the TD features are
kron(psi @ M, h(tau)), M a zone basis (`fork_td_basis`: 0 = all 63 zones, k = k
graph-Laplacian modes per zone-count group) and h = [1, cos, sin of the daily
phase, ... up to `fork_td_harmonics`].  Sub-models (fewer modes / harmonics,
`fork_td_sub`) are coordinate subsets and are solved from the same sums.
With `fork_td_extra` the TD features also include the per-car busy-time
features (busy-time buckets by number of trips, time-until-first-free buckets,
busy sums), which are confounded on-trajectory but not in fork differences.
Outputs forktdL{k}h{nh}[x]_{ridge}_H{H} (fork_H with theta_f) and ..._pair (paired
estimator with theta_f), for each sub-model and ridge.

Outputs (see `labels`): the PoolLSTDDQEstimator variants, then fork_H{H} for
each H in `fork_horizons` (and _jk versions if jackknife_blocks > 0), then
forkmc_H{H}, then forktd_* if fork_td, then diagnostics fkdiag_* (fraction of
forks closed early, dropped, mean ghosts).
"""
import numpy as np
from typing import Tuple
from flax import struct
import jax
import jax.numpy as jnp

from or_gymnax.rideshare_pool import obs_to_state, insert_and_optimize_trip
from xp_gym.estimators.lstd_dq_pool import PoolLSTDDQEstimator, PoolLSTDDQState, HI
from xp_gym.estimators.pool_features import pool_features

BIG = jnp.iinfo(jnp.int32).max


@struct.dataclass
class ForkDQState(PoolLSTDDQState):
    f_active: jnp.ndarray     # (G,) bool
    f_born: jnp.ndarray       # (G,) step index of the fork's decision
    f_sign: jnp.ndarray       # (G,) s_t
    f_cum: jnp.ndarray        # (G,) sum of (r_real - r_fork) so far
    f_n: jnp.ndarray          # (G,) number of ghosts
    f_car: jnp.ndarray        # (G, K) car ids (-1 = empty)
    f_W: jnp.ndarray          # (G, K, nwp) ghost waypoints
    f_T: jnp.ndarray          # (G, K, nwp) ghost times
    acc_r: jnp.ndarray        # (H,) sum s * cum at checkpoint
    acc_phi: jnp.ndarray      # (H, D1) sum s * (phi(fork) - phi(real)) at checkpoint
    n_open: jnp.ndarray       # forks opened
    n_early: jnp.ndarray      # forks closed early (ghost overflow)
    n_drop: jnp.ndarray       # differing steps with no free slot (H=1 fallback)
    ghost_sum: jnp.ndarray    # sum over steps of mean ghosts per active fork
    ghost_cnt: jnp.ndarray
    early_age: jnp.ndarray    # sum of fork ages at early close
    f_lpsi: jnp.ndarray       # (G, Dt) TD features at the start of the previous step
    f_ldr: jnp.ndarray        # (G,) r_real - r_fork at the previous step
    f_tdv: jnp.ndarray        # (G,) bool: previous-step transition continues
    td: dict                  # fork-TD sums (C, b, s1, S = sum psi psi^T, n); flushed from tdc
    tdc: dict                 # per-chunk fork-TD sums (float32 accuracy)
    acc_tphi: jnp.ndarray     # (H, Dt) like acc_phi, in TD features
    P_tphi: jnp.ndarray       # (Dt,) like P_phi, in TD features


@struct.dataclass
class ForkDQEstimator(PoolLSTDDQEstimator):
    fork_horizons: Tuple[int, ...] = struct.field(pytree_node=False, default=(1, 10, 30, 100, 300))
    fork_slots: int = struct.field(pytree_node=False, default=128)
    fork_ghosts: int = struct.field(pytree_node=False, default=12)
    fork_td: bool = struct.field(pytree_node=False, default=False)
    fork_td_ridges: Tuple[float, ...] = struct.field(pytree_node=False, default=(1e-6, 1e-3, 1e-1))
    # zone resolution of the fork-TD features: 0 = all 63 zones, k = first k graph-Laplacian
    # eigenvectors of the zone adjacency (zone_lap.npz), applied to each of the 3 zone-count groups
    fork_td_basis: int = struct.field(pytree_node=False, default=0)
    fork_td_harmonics: int = struct.field(pytree_node=False, default=0)    # daily Fourier pairs
    fork_td_extra: bool = struct.field(pytree_node=False, default=False)
    # sub-models (zone modes k, harmonics nh[, use extras 0/1])
    fork_td_sub: Tuple[Tuple[int, ...], ...] = struct.field(pytree_node=False, default=((0, 0),))
    day_seconds: float = struct.field(pytree_node=False, default=86400.0)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "fork_horizons", tuple(sorted(int(h) for h in self.fork_horizons)))
        assert self.fork_horizons[0] >= 1
        object.__setattr__(self, "fork_td_ridges", tuple(float(r) for r in self.fork_td_ridges))
        object.__setattr__(self, "fork_td_sub", tuple(
            (int(x[0]), int(x[1]), int(x[2]) if len(x) > 2 else int(self.fork_td_extra))
            for x in self.fork_td_sub))
        for k, nh, xx in self.fork_td_sub:
            assert nh <= self.fork_td_harmonics and (k <= self.fork_td_basis if self.fork_td_basis else k == 0)
            assert xx <= int(self.fork_td_extra)

    @property
    def labels(self):
        base = super().labels
        fk = [f"fork_H{h}" for h in self.fork_horizons]
        if self.jackknife_blocks:
            fk = fk + [f"{x}_jk" for x in fk]
        mc = [f"forkmc_H{h}" for h in self.fork_horizons]   # reward part only (U = 0)
        td = []
        if self.fork_td:
            for k, nh, xx in self.fork_td_sub:
                tag = f"L{k}h{nh}{'x' if xx else ''}_"
                for rg in self.fork_td_ridges:
                    td += [f"forktd{tag}{rg:g}_H{h}" for h in self.fork_horizons] + [f"forktd{tag}{rg:g}_pair"]
        return base + fk + mc + td + ["fkdiag_early", "fkdiag_drop", "fkdiag_ghosts", "fkdiag_early_age"]

    def reset(self, rng, env, env_params, design):
        st = super().reset(rng, env, env_params, design)
        G, K, H = self.fork_slots, self.fork_ghosts, len(self.fork_horizons)
        nwp = 2 * env_params.env_params.max_active_trips
        D1 = st.prev_phi.shape[0]
        Dt = self._td_dim(D1) if self.fork_td else 1
        z = lambda *s: jnp.zeros(s, jnp.float32)
        base = {f: getattr(st, f) for f in st.__dataclass_fields__}
        return ForkDQState(**base,
            f_active=jnp.zeros(G, bool), f_born=jnp.zeros(G, jnp.int32), f_sign=z(G), f_cum=z(G),
            f_n=jnp.zeros(G, jnp.int32), f_car=-jnp.ones((G, K), jnp.int32),
            f_W=jnp.zeros((G, K, nwp), jnp.int32), f_T=jnp.zeros((G, K, nwp), jnp.int32),
            acc_r=z(H), acc_phi=z(H, D1), n_open=z(), n_early=z(), n_drop=z(),
            ghost_sum=z(), ghost_cnt=z(), early_age=z(),
            f_lpsi=z(G, Dt), f_ldr=z(G), f_tdv=jnp.zeros(G, bool),
            td=self._td_zero(Dt), tdc=self._td_zero(Dt), acc_tphi=z(H, Dt), P_tphi=z(Dt))

    @staticmethod
    def _td_zero(D1):
        z = lambda *s: jnp.zeros(s, jnp.float32)
        return dict(C=z(D1, D1), b=z(D1), s1=z(D1), S=z(D1, D1), n=z())

    # ------------------------------------------------------------------
    @staticmethod
    def _extra_idx():
        from xp_gym.estimators.pool_features import feature_names
        names = feature_names(63)
        return np.asarray([i for i, nm in enumerate(names)
                           if nm.startswith(("busy1_", "busy2_", "firstfree2_"))
                           or nm in ("busy_sum", "busy_sq_sum", "firstfree_sum")])

    def _dpsi(self, setup, gW, gT, rW, rT, valid, tau):
        """phi(ghost cars) - phi(their real counterparts), zone features (+ intercept 0)
        [+ busy-time extras if fork_td_extra]."""
        ip, n2z, nz, idx, _ = setup
        # invalid slots: make both sides identical
        gW = jnp.where(valid[:, None], gW, rW)
        gT = jnp.where(valid[:, None], gT, rT)
        f = lambda W, T: pool_features(W, T, tau, n2z, nz)
        d = f(gW, gT) - f(rW, rT)
        out = [d[idx], jnp.zeros(1)]
        if self.fork_td and self.fork_td_extra:
            out.append(d[jnp.asarray(self._extra_idx())])
        return jnp.concatenate(out).astype(jnp.float32)

    def update(self, env, env_params, design, state, obs):
        new = super().update(env, env_params, design, state, obs)
        setup = self._setup(env_params)
        ip = setup[0]
        n_cars, nwp, mat = ip.n_cars, 2 * ip.max_active_trips, ip.max_active_trips
        G, K = self.fork_slots, self.fork_ghosts
        Hs = jnp.asarray(self.fork_horizons, jnp.int32)
        Hmax = self.fork_horizons[-1]

        ev, Wp, Tp = obs_to_state(n_cars, nwp, state.prev_obs)   # pre-decision fleet, request t
        Wp = Wp.astype(jnp.int32); Tp = Tp.astype(jnp.int32)
        t = state.t
        has_prev = t > 0
        z = obs.action.astype(jnp.int32)
        th = jnp.asarray(self.thresholds, jnp.float32)[z]
        r_real = obs.reward.astype(jnp.float32)
        cA = obs.info["action_A"].reshape(-1)[0].astype(jnp.int32)   # real (canonical) car
        cB = obs.info["action_B"].reshape(-1)[0].astype(jnp.int32)   # other arm's car
        direct = ip.distances[ev.src, ev.dest]
        fare = direct * (1 + ip.profit_margin)

        ins = lambda w, tt: insert_and_optimize_trip(ip.distances, w, tt, ev.src, ev.dest, ev.t, mat)

        def elig_cost(T, c, feas):
            solo = jnp.all(T <= ev.t, -1)
            ok = feas & (solo | (c < direct * (1 - th)))
            return jnp.where(ok, c, BIG)

        # ---- 1. checkpoints at the start of this step (state y_{t-1} at tau = ev.t)
        f_car = state.f_car
        gvalid = (f_car >= 0) & state.f_active[:, None]
        safe = jnp.maximum(f_car, 0)
        rW, rT = Wp[safe], Tp[safe]                                   # (G,K,nwp)
        dpsi_x = jax.vmap(lambda a, b, c, d, v: self._dpsi(setup, a, b, c, d, v, ev.t))(
            state.f_W, state.f_T, rW, rT, gvalid)                      # (G,D1[+E])
        D1 = state.acc_phi.shape[-1]
        dpsi = dpsi_x[:, :D1]
        age = t - state.f_born
        act = state.f_active & has_prev
        hit = act[:, None] & (age[:, None] == Hs[None, :])            # (G,H)
        sw = state.f_sign * state.f_cum
        acc_r = state.acc_r + jnp.sum(hit * sw[:, None], 0)
        acc_phi = state.acc_phi + jnp.einsum("gh,g,gd->hd", hit.astype(jnp.float32), state.f_sign, dpsi,
                                             precision=HI)
        active = act & (age < Hmax)

        # fork-TD transition  psi_{a-1} --dr--> psi_a  for forks alive at both steps
        tdc, td, acc_tphi = state.tdc, state.td, state.acc_tphi
        if self.fork_td:
            tdf = lambda x: self._td_feat(x, ev.t)
            tpsi = tdf(dpsi_x)                                         # (G, Dt)
            acc_tphi = acc_tphi + jnp.einsum("gh,g,gd->hd", hit.astype(jnp.float32), state.f_sign, tpsi,
                                             precision=HI)
            v = (state.f_tdv & act).astype(jnp.float32)
            lp = state.f_lpsi * v[:, None]
            inc = dict(C=jnp.einsum("gd,ge->de", lp, state.f_lpsi - tpsi, precision=HI),
                       b=-jnp.einsum("gd,g->d", lp, state.f_ldr, precision=HI),
                       s1=lp.sum(0), S=jnp.einsum("gd,ge->de", lp, lp, precision=HI), n=v.sum())
            tdc = {k: tdc[k] + inc[k] for k in tdc}
            flush = (t % 1024) == 0
            td = {k: jnp.where(flush, td[k] + tdc[k], td[k]) for k in td}
            tdc = {k: jnp.where(flush, jnp.zeros_like(tdc[k]), tdc[k]) for k in tdc}

        # ---- 2. replay this step in each open fork
        rw, rt, rc, rf = jax.vmap(ins)(Wp, Tp)                         # real cars, (n_cars,...)
        real_mc = elig_cost(Tp, rc, rf)
        gw, gt, gc, gf = jax.vmap(jax.vmap(ins))(state.f_W, state.f_T)  # ghosts, (G,K,...)
        g_mc = jnp.where(gvalid, elig_cost(state.f_T, gc, gf), BIG)
        # fork-world cost vector: real costs with ghost cars overwritten (dummy column n_cars)
        col = jnp.where(gvalid, f_car, n_cars)
        mc_f = jnp.broadcast_to(jnp.append(real_mc, BIG), (G, n_cars + 1))
        mc_f = jax.vmap(lambda m, c, v: m.at[c].set(v))(mc_f, col, g_mc)[:, :n_cars]
        c_f = jnp.argmin(mc_f, -1).astype(jnp.int32)
        found_f = jnp.min(mc_f, -1) < BIG
        cost_f = jnp.take_along_axis(mc_f, c_f[:, None], 1)[:, 0]
        r_f = jnp.where(found_f, fare - cost_f, 0.0).astype(jnp.float32)
        cum_new = state.f_cum + (r_real - r_f)

        is_g = gvalid & (f_car == c_f[:, None]) & found_f[:, None]       # cf car is a ghost
        cf_is_ghost = is_g.any(-1)
        a_is_ghost = (gvalid & (f_car == cA)).any(-1)
        add_Y = found_f & ~cf_is_ghost & (c_f != cA)
        add_X = (cA >= 0) & ~a_is_ghost & ~(found_f & (c_f == cA))
        n_new = state.f_n + add_X.astype(jnp.int32) + add_Y.astype(jnp.int32)
        overflow = active & (n_new > K)

        # early close: bootstrap at y_{t-1} for all checkpoints not yet reached
        late = overflow[:, None] & (Hs[None, :] > age[:, None])
        acc_r = acc_r + jnp.sum(late * sw[:, None], 0)
        acc_phi = acc_phi + jnp.einsum("gh,g,gd->hd", late.astype(jnp.float32), state.f_sign, dpsi,
                                       precision=HI)
        if self.fork_td:
            acc_tphi = acc_tphi + jnp.einsum("gh,g,gd->hd", late.astype(jnp.float32), state.f_sign, tpsi,
                                             precision=HI)
        active = active & ~overflow

        # apply the step to surviving forks
        f_W = jnp.where(is_g[..., None], gw, state.f_W)
        f_T = jnp.where(is_g[..., None], gt, state.f_T)
        slotX = state.f_n
        slotY = state.f_n + add_X.astype(jnp.int32)
        ks = jnp.arange(K)[None, :]
        wX = (add_X & active)[:, None] & (ks == slotX[:, None])
        wY = (add_Y & active)[:, None] & (ks == slotY[:, None])
        cAs = jnp.maximum(cA, 0)
        f_W = jnp.where(wX[..., None], Wp[cAs][None, None], f_W)
        f_T = jnp.where(wX[..., None], Tp[cAs][None, None], f_T)
        f_W = jnp.where(wY[..., None], rw[c_f][:, None], f_W)
        f_T = jnp.where(wY[..., None], rt[c_f][:, None], f_T)
        f_car = jnp.where(wX, cA, f_car)
        f_car = jnp.where(wY, c_f[:, None], f_car)
        f_n = jnp.where(active, n_new, 0)
        f_cum = jnp.where(active, cum_new, 0.0)
        f_lpsi = tpsi if self.fork_td else state.f_lpsi
        f_ldr, f_tdv = r_real - r_f, active

        # drop ghosts that re-merged with the real car: both idle at the same node
        # (real car's state after this step comes from the new observation)
        evn, Wn, Tn = obs_to_state(n_cars, nwp, obs.obs)
        Wn = Wn.astype(jnp.int32); Tn = Tn.astype(jnp.int32)
        tnext = evn.t.astype(jnp.int32)
        sc = jnp.maximum(f_car, 0)
        def last_node(W, T):
            return jnp.take_along_axis(W, jnp.argmax(T, -1)[..., None], -1)[..., 0]
        same = ((f_car >= 0) & jnp.all(f_T <= tnext, -1) & jnp.all(Tn[sc] <= tnext, -1)
                & (last_node(f_W, f_T) == last_node(Wn[sc], Tn[sc])))
        same = same | ((f_car >= 0) & jnp.all(f_W == Wn[sc], -1) & jnp.all(f_T == Tn[sc], -1))
        keep = (f_car >= 0) & ~same & active[:, None]
        order = jnp.argsort(~keep, axis=-1, stable=True)
        f_car = jnp.where(jnp.take_along_axis(keep, order, -1), jnp.take_along_axis(f_car, order, -1), -1)
        f_W = jnp.take_along_axis(f_W, order[..., None], 1)
        f_T = jnp.take_along_axis(f_T, order[..., None], 1)
        f_n = keep.sum(-1).astype(jnp.int32)

        # ---- 3. open a fork for this step if the arms differ
        d = (cA != cB) & has_prev
        free = ~active
        slot = jnp.argmax(free)
        can = d & free.any()
        drop = d & ~free.any()
        s_t = 2.0 * obs.action.astype(jnp.float32) - 1.0
        # immediate reward of the other arm
        cBs = jnp.maximum(cB, 0)
        foundB = cB >= 0
        r_cf = jnp.where(foundB, fare - rc[cBs], 0.0).astype(jnp.float32)
        imm = r_real - r_cf
        newcar = -jnp.ones(K, jnp.int32)
        newW = jnp.zeros((K, nwp), jnp.int32); newT = jnp.zeros((K, nwp), jnp.int32)
        hasA = cA >= 0
        newcar = newcar.at[0].set(jnp.where(hasA, cA, jnp.where(foundB, cB, -1)))
        newW = newW.at[0].set(jnp.where(hasA, Wp[cAs], rw[cBs]))
        newT = newT.at[0].set(jnp.where(hasA, Tp[cAs], rt[cBs]))
        both = hasA & foundB
        newcar = newcar.at[1].set(jnp.where(both, cB, -1))
        newW = newW.at[1].set(rw[cBs]); newT = newT.at[1].set(rt[cBs])
        nn = hasA.astype(jnp.int32) + foundB.astype(jnp.int32)
        upd = lambda arr, val: jnp.where(can, arr.at[slot].set(val), arr)
        active = upd(active, True)
        f_born = upd(state.f_born, t)
        f_sign = upd(state.f_sign, s_t)
        f_cum = upd(f_cum, imm)
        f_n = upd(f_n, nn)
        f_car = upd(f_car, newcar)
        f_W = upd(f_W, newW)
        f_T = upd(f_T, newT)
        f_tdv = upd(f_tdv, False)
        # no free slot: fall back to the paired (H=1) contribution for this step
        dphi_pair = new.P_phi - state.P_phi            # s d (phi_cf - phi) for this step
        acc_r = acc_r + jnp.where(drop, s_t * imm, 0.0)
        acc_phi = acc_phi + jnp.where(drop, dphi_pair, 0.0)[None, :]
        P_tphi = state.P_tphi
        if self.fork_td:
            # (extras of the paired fallback are not tracked: zero)
            xp = jnp.concatenate([dphi_pair, jnp.zeros(self._n_extra())])
            tpair = self._td_feat(xp[None, :], ev.t)[0]
            acc_tphi = acc_tphi + jnp.where(drop, tpair, 0.0)[None, :]
            P_tphi = P_tphi + tpair

        n_act = active.sum()
        return new.replace(
            f_active=active, f_born=f_born, f_sign=f_sign, f_cum=f_cum, f_n=f_n, f_car=f_car,
            f_W=f_W, f_T=f_T, acc_r=acc_r, acc_phi=acc_phi,
            n_open=state.n_open + can, n_early=state.n_early + overflow.sum(),
            n_drop=state.n_drop + drop,
            ghost_sum=state.ghost_sum + jnp.where(n_act > 0, f_n.sum() / jnp.maximum(n_act, 1), 0.0),
            ghost_cnt=state.ghost_cnt + (n_act > 0),
            early_age=state.early_age + jnp.sum(jnp.where(overflow, age, 0)),
            f_lpsi=f_lpsi, f_ldr=f_ldr, f_tdv=f_tdv, td=td, tdc=tdc, acc_tphi=acc_tphi, P_tphi=P_tphi)

    def _td_nh(self):
        return 1 + 2 * self.fork_td_harmonics

    def _n_extra(self):
        return len(self._extra_idx()) if self.fork_td_extra else 0

    def _td_dim(self, D1):
        return (self._td_basis(D1, self.fork_td_basis).shape[1] + self._n_extra()) * self._td_nh()

    def _td_feat(self, x, tau):
        """kron([x[:D1] @ M, x[D1:]], h(tau)) for rows x (.., D1 + n_extra)."""
        D1 = x.shape[-1] - self._n_extra()
        M = self._td_basis(D1, self.fork_td_basis)
        E = self._n_extra()
        Mx = np.zeros((D1 + E, M.shape[1] + E)); Mx[:D1, :M.shape[1]] = M; Mx[D1:, M.shape[1]:] = np.eye(E)
        M = jnp.asarray(Mx, jnp.float32)
        ph = 2 * jnp.pi * tau.astype(jnp.float32) / self.day_seconds
        h = [jnp.ones(())]
        for j in range(1, self.fork_td_harmonics + 1):
            h += [jnp.cos(j * ph), jnp.sin(j * ph)]
        h = jnp.stack(h)
        y = jnp.dot(x, M, precision=HI)
        return (y[..., :, None] * h).reshape(*x.shape[:-1], -1)

    def _td_select(self, D1, k, nh, xx=0):
        """Coordinates of the (k modes, nh harmonics, extras xx) sub-model in the TD features."""
        M = self._td_basis(D1, self.fork_td_basis)
        d, NH = M.shape[1], self._td_nh()
        K = self.fork_td_basis
        n0 = D1 - 1 - 3 * 63                      # leading non-zone features (ntrips)
        keep = []
        for j in range(d):
            if K and n0 <= j < d - 1 and (j - n0) % K >= k:
                continue
            keep += [j * NH + i for i in range(1 + 2 * nh)]
        if xx:
            for j in range(d, d + self._n_extra()):
                keep += [j * NH + i for i in range(1 + 2 * nh)]
        return np.asarray(keep)

    def _td_basis(self, D1, k):
        """(D1, d) projection: identity, or k Laplacian modes per zone-count group."""
        if not k:
            return np.eye(D1)
        import os
        V = np.load(os.path.join(os.path.dirname(__file__), "zone_lap.npz"))["V"][:, :k]
        nz = V.shape[0]
        n0 = D1 - 1 - 3 * nz                     # leading non-zone features (ntrips)
        M = np.zeros((D1, n0 + 3 * k + 1))
        M[:n0, :n0] = np.eye(n0)
        for g in range(3):
            M[n0 + g * nz:n0 + (g + 1) * nz, n0 + g * k:n0 + (g + 1) * k] = V
        M[-1, -1] = 1.0
        return M

    def theta_td(self, state, ridge, k=0, nh=0, xx=0):
        """Ridge LSTD on fork transitions (standardised, float64 on host), on the
        (k, nh) sub-model; returned in the full TD-feature space (zeros elsewhere)."""
        Dt = state.td["C"].shape[-1]
        sel = self._td_select(state.prev_phi.shape[-1], k, nh, xx)
        M = np.eye(Dt)[:, sel]

        def solve(C, b, s1, S, n, ridge):
            C, b, s1, S = (np.asarray(x, np.float64) for x in (C, b, s1, S))
            C, b, s1, S = M.T @ C @ M, M.T @ b, M.T @ s1, M.T @ S @ M
            n = max(float(n), 1.0)
            sd = np.sqrt(np.maximum(np.diag(S) / n - (s1 / n) ** 2, 0.0))
            sd = np.where(sd > 1e-6, sd, 1.0)
            A = C / n / np.outer(sd, sd)
            th = np.linalg.solve(A + float(ridge) * np.eye(len(A)), b / n / sd) / sd
            return (M @ th).astype(np.float32)

        def solve_batched(*args):
            C = np.asarray(args[0])
            if C.ndim == 2:
                return solve(*args)
            B = C.shape[0]
            bc = lambda x, i: np.broadcast_to(np.asarray(x), (B,) + np.asarray(x).shape[1:])[i]
            return np.stack([solve(*(bc(a, i) for a in args)) for i in range(B)])

        tot = {key: state.td[key] + state.tdc[key] for key in state.td}
        return jax.pure_callback(
            solve_batched, jax.ShapeDtypeStruct((Dt,), jnp.float32),
            tot["C"], tot["b"], tot["s1"], tot["S"], tot["n"], jnp.float32(ridge),
            vmap_method="expand_dims")

    def _fork_taus(self, state, th):
        Hs = jnp.asarray(self.fork_horizons, jnp.float32)
        Tk = jnp.maximum(state.t.astype(jnp.float32) - Hs, 1.0)
        return (state.acc_r - jnp.dot(state.acc_phi, th, precision=HI)) / Tk

    def estimate(self, env, env_params, design, state):
        base = super().estimate(env, env_params, design, state)
        th, _ = self.theta(state)
        fk = self._fork_taus(state, th)
        if self.jackknife_blocks:
            loo = jnp.stack([self._fork_taus(state, self.theta(state, k)[0])
                             for k in range(self.jackknife_blocks)])
            nonempty = state.blk["n"] > 0
            Ke = jnp.maximum(nonempty.sum(), 1).astype(jnp.float32)
            mean_loo = jnp.sum(jnp.where(nonempty[:, None], loo, 0.0), 0) / Ke
            fk = jnp.concatenate([fk, jnp.where(Ke > 1, Ke * fk - (Ke - 1) * mean_loo, fk)])
        Tk = jnp.maximum(state.t.astype(jnp.float32) - jnp.asarray(self.fork_horizons, jnp.float32), 1.0)
        fk = jnp.concatenate([fk, state.acc_r / Tk])
        if self.fork_td:
            # theta_f values V(real) - V(fork) = -theta_f . psi, the same sign
            # convention as the LSTD tail, so fork_H / paired formulas apply as is
            T = jnp.maximum(state.t, 1).astype(jnp.float32)
            Hs = jnp.asarray(self.fork_horizons, jnp.float32)
            Tk = jnp.maximum(state.t.astype(jnp.float32) - Hs, 1.0)
            for k, nh, xx in self.fork_td_sub:
                for rg in self.fork_td_ridges:
                    thf = self.theta_td(state, rg, k, nh, xx)
                    pair = (state.P_r - jnp.dot(thf, state.P_tphi, precision=HI)) / T
                    taus = (state.acc_r - jnp.dot(state.acc_tphi, thf, precision=HI)) / Tk
                    fk = jnp.concatenate([fk, taus, pair[None]])
        nd = jnp.maximum(state.n_open + state.n_drop, 1.0)
        diag = jnp.stack([state.n_early / jnp.maximum(state.n_open, 1.0), state.n_drop / nd,
                          state.ghost_sum / jnp.maximum(state.ghost_cnt, 1.0),
                          state.early_age / jnp.maximum(state.n_early, 1.0)])
        return jnp.concatenate([base, fk, diag])
