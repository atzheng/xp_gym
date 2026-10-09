"""Finite-sample vs projection bias: pure LSTD-DQ (zones features) with theta fitted
on 1, 2, 4, 8 pooled envs, evaluated on every env.
usage: python scratch/fs.py <dev dir> <out.csv>
"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "scratch")
from offline2 import env_files, groups
def build(X, fns):
    return np.hstack([X] + [f(X) for f in fns] + [np.ones((len(X), 1))])


def stats(X, fns, r, chunk=20000):
    N = len(r)
    D1 = build(X[:1], fns).shape[1]
    Cd = np.zeros((D1, D1)); b = np.zeros(D1); s1 = np.zeros(D1); s2 = np.zeros(D1)
    r1 = r[1:]; rbar = r1.mean()
    for i in range(0, N - 1, chunk):
        j = min(i + chunk, N - 1)
        Z = build(X[i:j + 1], fns)
        X0, X1 = Z[:-1], Z[1:]
        Cd += X0.T @ (X0 - X1); b += X0.T @ (r1[i:j] - rbar)
        s1 += X0.sum(0); s2 += (X0 ** 2).sum(0)
    n = N - 1
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))
    return dict(Cd=Cd, b=b, n=n, sd=sd)


def solve(S, ridge=1e-6):
    sd = np.where(S["sd"] > 1e-6, S["sd"], 1.0); sd[-1] = 1.0
    A = S["Cd"] / S["n"] / np.outer(sd, sd)
    return np.linalg.solve(A + ridge * np.eye(len(A)), S["b"] / S["n"] / sd) / sd



d, out = sys.argv[1], sys.argv[2]
from scipy.cluster.vq import kmeans2
zones = pd.read_parquet('xp_gym/data/taxi-zones.parquet')
nodes = pd.read_parquet('xp_gym/data/manhattan-nodes.parquet').astype({'lat': float, 'lng': float, 'osmid': str})
zones['osmid'] = zones.osmid.astype(str)
_, zid = np.unique(zones['zone'], return_inverse=True); zones['zid'] = zid
cent = nodes.merge(zones, on='osmid').groupby('zid')[['lat', 'lng']].mean().sort_index().values
MAPS = {R: kmeans2(cent, R, seed=1, minit='++')[1] for R in [4, 8, 16, 32]}
MAPS[63] = np.arange(63); MAPS[0] = None
def tf(X, R):
    if MAPS[R] is None: return X[:, :3]
    M = np.zeros((63, R)); M[np.arange(63), MAPS[R]] = 1
    return np.hstack([X[:, :3]] + [X[:, 3 + 63 * k:3 + 63 * (k + 1)] @ M for k in range(3)])
envs, St = {}, {}
for ei, f in enumerate(env_files(d)):
    x = np.load(f)
    g = groups(x["names"])
    cols = g["ntrips"] + g["idle"] + g["one_final"] + g["full_drop"]
    r = x["r"].astype(np.float64); N = len(r) - 1; r = r[:N]
    X = x["phi"][:N][:, cols].astype(np.float64)
    didx = x["didx"]; keep = didx < N
    dX = x["dphi"][keep][:, cols].astype(np.float64); didx = didx[keep]
    z = x["z"][:N].astype(np.float64); rcf = x["r_cf"][:N].astype(np.float64)
    s = 2 * z[didx] - 1
    envs[ei] = dict(N=N, imm=np.sum(s * (r[didx] - rcf[didx])))
    for R in MAPS:
        XR = tf(X, R)
        envs[ei][R] = np.append((s[:, None] * tf(dX, R)).sum(0), 0.0)
        St[(R, ei)] = stats(XR, [], r)
        St[(R, ei, 0)] = stats(XR[:N // 2], [], r[:N // 2])
        St[(R, ei, 1)] = stats(XR[N // 2:], [], r[N // 2:])
    del X, x
    print("env", ei, flush=True)

E = len(envs)
rows = []
def pool(keys):
    P = {k: sum(St[kk][k] for kk in keys) for k in ["Cd", "b", "n"]}
    P["sd"] = np.mean([St[kk]["sd"] for kk in keys], 0)
    return solve(P)
for R in MAPS:
    evals = lambda th: np.mean([(e["imm"] - e[R] @ th) / e["N"] for e in envs.values()])
    for h in range(2):
        for ei in range(E):
            rows.append(dict(R=R, n_half_envs=1, est=evals(pool([(R, ei, h)]))))
    for m in [1, 2, 4, 8]:
        for start in range(0, E, m):
            rows.append(dict(R=R, n_half_envs=2 * m, est=evals(pool([(R, k) for k in range(start, start + m)]))))
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(df.groupby(["R", "n_half_envs"]).est.agg(["mean", "std", "count"]).round(2).unstack(0).to_string())
