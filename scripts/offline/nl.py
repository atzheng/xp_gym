"""Offline test: do nonlinear count features (diminishing returns / substitution)
reduce pure LSTD-DQ projection bias?
usage: python scratch/nl.py <dev dir> <out.csv>
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch")
from offline2 import env_files, groups

d, out = sys.argv[1], sys.argv[2]
# k-means regions of the 63 zones (from scratch/coarse.py if available)
try:
    from coarse import region_map
except Exception:
    region_map = None


def make_fsets(g):
    nt = g["ntrips"]; zb = g["idle"] + g["one_final"] + g["full_drop"]
    zones = nt + zb
    # transforms are functions of the zones block X (columns in `zones` order)
    k = len(nt)
    def glob2(X):  # global counts, squares and pairwise products
        c = X[:, :k]
        return np.hstack([c ** 2 / 300.0, (c[:, [0]] * c[:, [1]]) / 300.0,
                          (c[:, [0]] * c[:, [2]]) / 300.0, (c[:, [1]] * c[:, [2]]) / 300.0])
    def zsqrt(X): return np.sqrt(X[:, k:])
    def zsq(X): return X[:, k:] ** 2
    def ztot(X):  # per-zone totals of idle + 1-trip (pool-able supply) and their sqrt
        nz = (X.shape[1] - k) // 3
        sup = X[:, k:k + nz] + X[:, k + nz:k + 2 * nz]
        return np.sqrt(sup)
    F = {"zones": [], "zones+glob2": [glob2], "zones+zsqrt": [zsqrt], "zones+zsq": [zsq],
         "zones+ztot": [ztot], "zones+glob2+zsqrt": [glob2, zsqrt]}
    return zones, F


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


envs, St = {}, {}
for ei, f in enumerate(env_files(d)):
    x = np.load(f)
    g = groups(x["names"])
    cols, F = make_fsets(g)
    r = x["r"].astype(np.float64); N = len(r) - 1; r = r[:N]
    X = x["phi"][:N][:, cols].astype(np.float64)
    didx = x["didx"]; keep = didx < N
    dX = x["dphi"][keep][:, cols].astype(np.float64); didx = didx[keep]
    z = x["z"][:N].astype(np.float64); rcf = x["r_cf"][:N].astype(np.float64)
    s = 2 * z[didx] - 1
    # phi_cf = phi + dphi at the differing steps
    Xd, Xc = X[didx], X[didx] + dX
    imm = np.sum(s * (r[didx] - rcf[didx]))
    envs[ei] = dict(N=N, imm=imm, dv={})
    for fn, fns in F.items():
        St[(fn, ei)] = stats(X, fns, r)
        envs[ei]["dv"][fn] = (s[:, None] * (build(Xc, fns) - build(Xd, fns))).sum(0)
    del X, x
    print("env", ei, flush=True)

rows = []
for fn in F:
    keys = [k for k in St if k[0] == fn]
    P = {k: sum(St[kk][k] for kk in keys) for k in ["Cd", "b", "n"]}
    P["sd"] = np.mean([St[kk]["sd"] for kk in keys], 0)
    thp = solve(P)
    for (_, ei) in keys:
        e = envs[ei]; tho = solve(St[(fn, ei)])
        rows.append(dict(f=fn, env=ei, D=len(thp), pooled=(e["imm"] - e["dv"][fn] @ thp) / e["N"],
                         own=(e["imm"] - e["dv"][fn] @ tho) / e["N"]))
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(df.groupby("f", sort=False)[["D", "pooled", "own"]].agg(["mean", "std"]).round(2))
