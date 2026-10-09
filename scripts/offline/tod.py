"""Offline test: do time-of-day / demand-rate features reduce pure LSTD-DQ bias?
usage: python scratch/tod.py <dev dir> <out.csv>
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch")
from offline2 import env_files, groups

d, out = sys.argv[1], sys.argv[2]
T_EV = np.load("scratch/event_t.npy").astype(np.float64)
N_EV = len(T_EV)
# phi row i is evaluated at the next request's time
tt = np.concatenate([T_EV[1:], T_EV[-1:]])
ph = 2 * np.pi * (tt % 86400) / 86400
# trailing 30-min request rate, normalised
cnt = np.arange(N_EV) - np.searchsorted(T_EV, T_EV - 1800, side="left")
rate = np.concatenate([cnt[1:], cnt[-1:]]).astype(np.float64)
rate /= rate.mean()
day = tt / 86400 / 7
TIME = np.stack([np.sin(k * ph) for k in range(1, 5)] + [np.cos(k * ph) for k in range(1, 5)]
                + [day, rate, rate ** 2], 1)  # time-only features
G = {"none": np.zeros((N_EV, 0)),
     "tod1": np.stack([np.sin(ph), np.cos(ph)], 1),
     "tod2": np.stack([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)], 1),
     "rate": rate[:, None],
     "rate2": np.stack([rate, rate ** 2], 1)}
FSETS = [("zones", False, "none"), ("zones+T", True, "none"), ("zones+T+tod1", True, "tod1"),
         ("zones+T+tod2", True, "tod2"), ("zones+T+rate", True, "rate"), ("zones+T+rate2", True, "rate2")]


def build(base, i, j, use_t, g):
    parts = [base[i:j]]
    if use_t:
        parts.append(TIME[i:j])
    gg = G[g][i:j]
    for k in range(gg.shape[1]):
        parts.append(base[i:j] * gg[:, k:k + 1])
    parts.append(np.ones((j - i, 1)))
    return np.hstack(parts)


def build_d(dbase, idx, use_t, g):
    parts = [dbase]
    if use_t:
        parts.append(np.zeros((len(idx), TIME.shape[1])))
    gg = G[g][idx]
    for k in range(gg.shape[1]):
        parts.append(dbase * gg[:, k:k + 1])
    parts.append(np.zeros((len(idx), 1)))
    return np.hstack(parts)


def stats(base, r, use_t, g, chunk=20000):
    N = len(r)
    D1 = build(base, 0, 1, use_t, g).shape[1]
    Cd = np.zeros((D1, D1)); b = np.zeros(D1); s1 = np.zeros(D1); s2 = np.zeros(D1)
    r1 = r[1:]; rbar = r1.mean()
    for i in range(0, N - 1, chunk):
        j = min(i + chunk, N - 1)
        X = build(base, i, j + 1, use_t, g)
        X0, X1 = X[:-1], X[1:]
        Cd += X0.T @ (X0 - X1); b += X0.T @ (r1[i:j] - rbar)
        s1 += X0.sum(0); s2 += (X0 ** 2).sum(0)
    n = N - 1
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))
    return dict(Cd=Cd, b=b, n=n, sd=sd)


def solve(S, ridge=1e-6):
    sd = np.where(S["sd"] > 1e-6, S["sd"], 1.0); sd[-1] = 1.0
    A = S["Cd"] / S["n"] / np.outer(sd, sd)
    return np.linalg.solve(A + ridge * np.eye(len(A)), S["b"] / S["n"] / sd) / sd


envs = {}
St = {}
for ei, f in enumerate(env_files(d)):
    x = np.load(f)
    names = x["names"]; g = groups(names)
    cols = g["ntrips"] + g["idle"] + g["one_final"] + g["full_drop"]
    r = x["r"].astype(np.float64)
    N = len(r) - 1
    r = r[:N]
    base = x["phi"][:N][:, cols].astype(np.float64)
    didx = x["didx"]; keep = didx < N
    dbase = x["dphi"][keep][:, cols].astype(np.float64)
    didx = didx[keep]
    z = x["z"][:N].astype(np.float64); rcf = x["r_cf"][:N].astype(np.float64)
    envs[ei] = dict(N=N, didx=didx, z=z, r=r, rcf=rcf, dX={})
    for fn, use_t, gname in FSETS:
        St[(fn, ei)] = stats(base, r, use_t, gname)
        envs[ei]["dX"][fn] = ((2 * z[didx] - 1)[:, None] * build_d(dbase, didx, use_t, gname)).sum(0)
    del base, x
    print("env", ei, flush=True)

rows = []
for fn, _, _ in FSETS:
    keys = [k for k in St if k[0] == fn]
    P = {k: sum(St[kk][k] for kk in keys) for k in ["Cd", "b", "n"]}
    P["sd"] = np.mean([St[kk]["sd"] for kk in keys], 0)
    thp = solve(P)
    for (_, ei) in keys:
        e = envs[ei]; i = e["didx"]; s = 2 * e["z"] - 1
        imm = np.sum(s[i] * (e["r"][i] - e["rcf"][i]))
        tho = solve(St[(fn, ei)])
        rows.append(dict(f=fn, env=ei, D=len(thp),
                         pooled=(imm - e["dX"][fn] @ thp) / e["N"],
                         own=(imm - e["dX"][fn] @ tho) / e["N"]))
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(df.groupby("f", sort=False)[["D", "pooled", "own"]].agg(["mean", "std"]).round(2))
