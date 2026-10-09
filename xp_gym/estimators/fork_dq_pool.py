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

Outputs (see `labels`): the PoolLSTDDQEstimator variants, then fork_H{H} for
each H in `fork_horizons` (and _jk versions if jackknife_blocks > 0), then
diagnostics fkdiag_* (fraction of forks closed early, dropped, mean ghosts).
"""
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


@struct.dataclass
class ForkDQEstimator(PoolLSTDDQEstimator):
    fork_horizons: Tuple[int, ...] = struct.field(pytree_node=False, default=(1, 10, 30, 100, 300))
    fork_slots: int = struct.field(pytree_node=False, default=128)
    fork_ghosts: int = struct.field(pytree_node=False, default=12)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "fork_horizons", tuple(sorted(int(h) for h in self.fork_horizons)))
        assert self.fork_horizons[0] >= 1

    @property
    def labels(self):
        base = super().labels
        fk = [f"fork_H{h}" for h in self.fork_horizons]
        if self.jackknife_blocks:
            fk = fk + [f"{x}_jk" for x in fk]
        mc = [f"forkmc_H{h}" for h in self.fork_horizons]   # reward part only (U = 0)
        return base + fk + mc + ["fkdiag_early", "fkdiag_drop", "fkdiag_ghosts", "fkdiag_early_age"]

    def reset(self, rng, env, env_params, design):
        st = super().reset(rng, env, env_params, design)
        G, K, H = self.fork_slots, self.fork_ghosts, len(self.fork_horizons)
        nwp = 2 * env_params.env_params.max_active_trips
        D1 = st.prev_phi.shape[0]
        z = lambda *s: jnp.zeros(s, jnp.float32)
        base = {f: getattr(st, f) for f in st.__dataclass_fields__}
        return ForkDQState(**base,
            f_active=jnp.zeros(G, bool), f_born=jnp.zeros(G, jnp.int32), f_sign=z(G), f_cum=z(G),
            f_n=jnp.zeros(G, jnp.int32), f_car=-jnp.ones((G, K), jnp.int32),
            f_W=jnp.zeros((G, K, nwp), jnp.int32), f_T=jnp.zeros((G, K, nwp), jnp.int32),
            acc_r=z(H), acc_phi=z(H, D1), n_open=z(), n_early=z(), n_drop=z(),
            ghost_sum=z(), ghost_cnt=z(), early_age=z())

    # ------------------------------------------------------------------
    def _dpsi(self, setup, gW, gT, rW, rT, valid, tau):
        """phi(ghost cars) - phi(their real counterparts), zone features (+ intercept 0)."""
        ip, n2z, nz, idx, _ = setup
        # invalid slots: make both sides identical
        gW = jnp.where(valid[:, None], gW, rW)
        gT = jnp.where(valid[:, None], gT, rT)
        f = lambda W, T: pool_features(W, T, tau, n2z, nz)[idx]
        d = f(gW, gT) - f(rW, rT)
        return jnp.concatenate([d, jnp.zeros(1)]).astype(jnp.float32)

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
        dpsi = jax.vmap(lambda a, b, c, d, v: self._dpsi(setup, a, b, c, d, v, ev.t))(
            state.f_W, state.f_T, rW, rT, gvalid)                      # (G,D1)
        age = t - state.f_born
        act = state.f_active & has_prev
        hit = act[:, None] & (age[:, None] == Hs[None, :])            # (G,H)
        sw = state.f_sign * state.f_cum
        acc_r = state.acc_r + jnp.sum(hit * sw[:, None], 0)
        acc_phi = state.acc_phi + jnp.einsum("gh,g,gd->hd", hit.astype(jnp.float32), state.f_sign, dpsi,
                                             precision=HI)
        active = act & (age < Hmax)

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
        # no free slot: fall back to the paired (H=1) contribution for this step
        dphi_pair = new.P_phi - state.P_phi            # s d (phi_cf - phi) for this step
        acc_r = acc_r + jnp.where(drop, s_t * imm, 0.0)
        acc_phi = acc_phi + jnp.where(drop, dphi_pair, 0.0)[None, :]

        n_act = active.sum()
        return new.replace(
            f_active=active, f_born=f_born, f_sign=f_sign, f_cum=f_cum, f_n=f_n, f_car=f_car,
            f_W=f_W, f_T=f_T, acc_r=acc_r, acc_phi=acc_phi,
            n_open=state.n_open + can, n_early=state.n_early + overflow.sum(),
            n_drop=state.n_drop + drop,
            ghost_sum=state.ghost_sum + jnp.where(n_act > 0, f_n.sum() / jnp.maximum(n_act, 1), 0.0),
            ghost_cnt=state.ghost_cnt + (n_act > 0),
            early_age=state.early_age + jnp.sum(jnp.where(overflow, age, 0)))

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
        nd = jnp.maximum(state.n_open + state.n_drop, 1.0)
        diag = jnp.stack([state.n_early / jnp.maximum(state.n_open, 1.0), state.n_drop / nd,
                          state.ghost_sum / jnp.maximum(state.ghost_cnt, 1.0),
                          state.early_age / jnp.maximum(state.n_early, 1.0)])
        return jnp.concatenate([base, fk, diag])
