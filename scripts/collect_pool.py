"""Simulate the unit-randomized p=0.5 experiment and dump per-step data for
offline DQ/LSTD development.

For each env writes <outdir>/env{e}.npz with
  z, r, differ, r_cf, unfulfill : (N,)
  phi   : (N, D) features of post-decision state y_t (evaluated at next request time)
  dphi  : (n_differ, D) phi(y_t^cf) - phi(y_t) on steps where the arms differ
  didx  : (n_differ,) step index of those rows

usage: python scripts/collect_pool.py outdir n_envs n_steps stb [seed] [K]
(stb may be "A:B" to set arm A's threshold too)
(K>0 adds match_features over K reference requests; outdir may be s3://)
"""
import os, sys, time
import jax, jax.numpy as jnp, numpy as np
from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.environments.rideshare_pool import insert_and_optimize_trip
from xp_gym.estimators.pool_features import (pool_features, load_node_to_zone, feature_names,
    reference_requests, match_costs, match_features_from_costs, match_feature_names)

outdir = sys.argv[1]; E = int(sys.argv[2]); N = int(sys.argv[3])
sta, stb = (float(x) for x in sys.argv[4].split(":")) if ":" in sys.argv[4] else (0.0, float(sys.argv[4]))
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
K = int(sys.argv[6]) if len(sys.argv) > 6 else 0
# match features at tau + offset (seconds); later offsets use the first K2 refs
OFFSETS = (0, 120, 300)
K2 = min(K, 64)
remote = outdir if outdir.startswith("s3://") else None
if remote: outdir = "/tmp/collect_out"
CHUNK = 10000
os.makedirs(outdir, exist_ok=True)

env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_A=sta,
                                 savings_threshold_B=stb)
p0 = env.default_params
params = p0.replace(env_params=p0.env_params.replace(
    max_active_trips=2, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))
ip = params.env_params
n2z, NZ = load_node_to_zone()
names = feature_names(NZ)
TH = (sta, stb)
if K:
    ref_src, ref_dest = reference_requests(ip.events, K)
    for o in OFFSETS:
        kk = K if o == 0 else K2
        names = names + [f"o{o}_{n}" for n in match_feature_names(kk, TH)]


def feats(W, T, tau, changed=None, base_costs=None):
    """Features of fleet (W, T) at tau. If base_costs is given, the fleet
    differs from the base fleet only in the cars listed in `changed`, and
    only those columns of the match-cost matrices are recomputed."""
    f = pool_features(W, T, tau, n2z, NZ)
    if not K:
        return f, None
    all_costs, out = [], [f]
    for i, o in enumerate(OFFSETS):
        kk = K if o == 0 else K2
        rs_, rd_ = ref_src[:kk], ref_dest[:kk]
        if base_costs is None:
            costs = match_costs(W, T, tau + o, ip.distances, rs_, rd_)
        else:
            sub = match_costs(W[changed], T[changed], tau + o, ip.distances, rs_, rd_)
            costs = tuple(b.at[:, changed].set(x) for b, x in zip(base_costs[i], sub))
        all_costs.append(costs)
        out.append(match_features_from_costs(*costs, ip.distances, rs_, rd_, TH))
    return jnp.concatenate(out), all_costs
D = len(names)


def step(s, k):
    k1, k2 = jax.random.split(k)
    z = jax.random.bernoulli(k1, 0.5).astype(jnp.int32)
    _, s2, r, _, info = env.step(k2, s, z, params)
    tau = s2.event.t
    phi, costs = feats(s2.waypoints, s2.times, tau)
    c_cf = info["action_B"]
    ev = s.event
    wp_cf, t_cf, mc_cf, _ = insert_and_optimize_trip(
        ip.distances, s.waypoints[jnp.maximum(c_cf, 0)], s.times[jnp.maximum(c_cf, 0)],
        ev.src, ev.dest, ev.t, ip.max_active_trips)
    found = c_cf >= 0
    W = jnp.where(found, s.waypoints.at[c_cf].set(wp_cf), s.waypoints)
    T = jnp.where(found, s.times.at[c_cf].set(t_cf), s.times)
    changed = jnp.stack([jnp.maximum(info["action_A"], 0), jnp.maximum(c_cf, 0)])
    phi_cf, _ = feats(W, T, tau, changed, costs)
    r_cf = jnp.where(found, ip.distances[ev.src, ev.dest] * (1 + ip.profit_margin) - mc_cf, 0.0)
    differ = info["action_A"] != info["action_B"]
    return s2, (z.astype(jnp.int8), r.astype(jnp.float32), differ, r_cf.astype(jnp.float32),
                jnp.asarray(info["is_unfulfill"]), phi.astype(jnp.float16), phi_cf - phi)


@jax.jit
@jax.vmap
def chunk(s, keys):
    return jax.lax.scan(step, s, keys)


keys = jax.random.split(jax.random.PRNGKey(seed), E)
_, s = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
step_keys = jax.vmap(lambda k: jax.random.split(k, N))(keys)
out = {k: [[] for _ in range(E)] for k in ["z", "r", "differ", "r_cf", "unfulfill", "phi", "dphi", "didx"]}
t0 = time.time()
for c in range(N // CHUNK):
    s, o = chunk(s, step_keys[:, c * CHUNK:(c + 1) * CHUNK])
    o = jax.tree.map(np.asarray, o)
    for e in range(E):
        for k, v in zip(["z", "r", "differ", "r_cf", "unfulfill", "phi"], o[:6]):
            out[k][e].append(v[e])
        d = o[2][e]
        out["dphi"][e].append(o[6][e][d])
        out["didx"][e].append(np.nonzero(d)[0] + c * CHUNK)
    if c % 5 == 0:
        print(f"chunk {c} {time.time()-t0:.0f}s", flush=True)
for e in range(E):
    np.savez(os.path.join(outdir, f"env{e}.npz"), names=np.array(names),
             **{k: np.concatenate(v[e]) for k, v in out.items()})
if remote:
    import s3fs
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("AWS_ENDPOINT_URL")})
    fs.put(outdir + "/", remote.rstrip("/") + "/", recursive=True)
print("done", time.time() - t0)
