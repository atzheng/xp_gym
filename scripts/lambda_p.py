"""Average reward lambda(p) of the p-mixture policy (unit-randomized), with
common random numbers across p.  DQ's estimand is lambda'(0.5); the ATE is
lambda(1) - lambda(0).

usage: python scripts/lambda_p.py out.csv n_envs n_steps stb [mat] [ps...]
(the event table is always the first 500k events, matching dq_expts)
"""
import sys, time
import jax, jax.numpy as jnp, numpy as np, pandas as pd
from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from xp_gym.io import to_csv

out = sys.argv[1]; E = int(sys.argv[2]); N = int(sys.argv[3]); stb = float(sys.argv[4])
mat = int(sys.argv[5]) if len(sys.argv) > 5 else 2
ps = [float(x) for x in sys.argv[6:]] or [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
CHUNK = 10000

env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_A=0.0,
                                 savings_threshold_B=stb)
p0 = env.default_params
params = p0.replace(env_params=p0.env_params.replace(
    max_active_trips=mat, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))


def chunk(p, carry, keys):
    def f(s, k):
        k1, k2 = jax.random.split(k)
        z = (jax.random.uniform(k1) < p).astype(jnp.int32)
        _, s, r, _, info = env.step(k2, s, z, params)
        d = (info["action_A"] != info["action_B"]).astype(jnp.float32)
        return s, jnp.stack([r, d, info["is_unfulfill"].astype(jnp.float32),
                             r * z, r * (1 - z), z.astype(jnp.float32)])
    s, o = jax.lax.scan(f, carry, keys)
    return s, o.sum(0)

vchunk = jax.jit(jax.vmap(chunk, in_axes=(None, 0, 0)))
rows = []
for p in ps:
    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(0), E)
    _, s = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    step_keys = jax.vmap(lambda k: jax.random.split(k, N))(keys)
    acc = []
    for c in range(N // CHUNK):
        s, o = vchunk(p, s, step_keys[:, c * CHUNK:(c + 1) * CHUNK])
        acc.append(np.asarray(o))
    acc = np.stack(acc, 1)  # E, n_chunks, 6
    for e in range(E):
        for c in range(acc.shape[1]):
            rows.append(dict(p=p, env=e, chunk=c, reward=acc[e, c, 0] / CHUNK,
                             differ=acc[e, c, 1] / CHUNK, unfulfill=acc[e, c, 2] / CHUNK,
                             rz1=acc[e, c, 3], rz0=acc[e, c, 4], nz1=acc[e, c, 5]))
    print(f"p={p} reward={acc[:,:,0].sum()/E/N:.4f} ({time.time()-t0:.0f}s)", flush=True)

to_csv(pd.DataFrame(rows), out)
