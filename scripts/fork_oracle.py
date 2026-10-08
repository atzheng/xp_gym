"""Ground-truth Q-differences at differing decisions via forked rollouts.

For each env: warm up W steps of the p=0.5 experiment, then repeatedly
(advance G steps; advance until arms differ; fork into A-branch / B-branch;
roll both out H steps with identical randomness under p=0.5).
Records cumulative reward difference (B - A) at checkpoints, the immediate
rewards, and post-decision features of both branches.

usage: python scripts/fork_oracle.py out.npz n_envs n_forks W G H stb [seed]
"""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.environments.rideshare_pool import greedy_select_car
from xp_gym.estimators.pool_features import pool_features, load_node_to_zone, feature_names

out = sys.argv[1]; E, K, W, G, H = map(int, sys.argv[2:7]); stb = float(sys.argv[7])
seed = int(sys.argv[8]) if len(sys.argv) > 8 else 1
CKPT = np.array([10, 30, 100, 300, 1000, 2000, 3000, 5000, 10000, 20000, 40000])
CKPT = CKPT[CKPT <= H]

env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_A=0.0,
                                 savings_threshold_B=stb)
p0 = env.default_params
params = p0.replace(env_params=p0.env_params.replace(
    max_active_trips=2, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))
ip = params.env_params
n2z, NZ = load_node_to_zone()


def step(s, k):
    k1, k2 = jax.random.split(k)
    z = jax.random.bernoulli(k1, 0.5).astype(jnp.int32)
    _, s2, r, _, _ = env.step(k2, s, z, params)
    return s2, r


def run(s, key, n):
    return jax.lax.scan(step, s, jax.random.split(key, n))


def differs(s):
    ca, fa = greedy_select_car(ip.distances, s.waypoints, s.times, s.event, 2, 0.0)
    cb, fb = greedy_select_car(ip.distances, s.waypoints, s.times, s.event, 2, stb)
    return (jnp.where(fa, ca, -1) != jnp.where(fb, cb, -1))


def advance_to_differ(s, key):
    def cond(c):
        s, k, n = c
        return (~differs(s)) & (n < 1000)
    def body(c):
        s, k, n = c
        k, k1 = jax.random.split(k)
        s, _ = step(s, k1)
        return s, k, n + 1
    s, _, _ = jax.lax.while_loop(cond, body, (s, key, 0))
    return s


def fork(s, key):
    kd, kr = jax.random.split(key)
    _, sA, rA, _, _ = env.step(kd, s, 0, params)
    _, sB, rB, _, _ = env.step(kd, s, 1, params)
    phiA = pool_features(sA.waypoints, sA.times, sA.event.t, n2z, NZ)
    phiB = pool_features(sB.waypoints, sB.times, sB.event.t, n2z, NZ)
    keys = jax.random.split(kr, H)
    _, rsA = jax.lax.scan(step, sA, keys)
    _, rsB = jax.lax.scan(step, sB, keys)
    cum = jnp.cumsum(rsB - rsA)[CKPT - 1]
    return dict(rA=rA, rB=rB, cum=cum, phiA=phiA, phiB=phiB, t=s.time)


@jax.jit
@jax.vmap
def warm(key):
    _, s = env.reset(key, params)
    s, _ = run(s, key, W)
    return s


@jax.jit
@jax.vmap
def one(s, key):
    k1, k2, k3 = jax.random.split(key, 3)
    s, _ = run(s, k1, G)
    s = advance_to_differ(s, k2)
    res = fork(s, k3)
    # continue main trajectory past the fork point
    s, _ = step(s, k3)
    return s, res


t0 = time.time()
keys = jax.random.split(jax.random.PRNGKey(seed), E)
s = warm(keys)
print("warm", time.time() - t0, flush=True)
res = []
for k in range(K):
    s, r = one(s, jax.vmap(lambda kk: jax.random.fold_in(kk, k))(keys))
    res.append(jax.tree.map(np.asarray, r))
    print("fork", k, time.time() - t0, flush=True)
res = {k: np.stack([r[k] for r in res], 1) for k in res[0]}
from xp_gym.io import to_csv  # noqa (for s3 handling below)
import io, os
buf = io.BytesIO(); np.savez(buf, ckpt=CKPT, names=np.array(feature_names(NZ)), **res)
if out.startswith("s3://"):
    import s3fs
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("AWS_ENDPOINT_URL")})
    with fs.open(out, "wb") as f: f.write(buf.getvalue())
else:
    open(out, "wb").write(buf.getvalue())
print("done", time.time() - t0)
