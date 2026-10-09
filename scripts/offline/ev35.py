"""Dev-set evaluation of LSTD-DQ variants on one dev dir (own theta per env, and pooled theta).
usage: python scratch/ev35.py <dev dir> <truth> <out.csv>
"""
import sys, numpy as np, pandas as pd
from scipy.signal import lfilter
sys.path.insert(0, "scratch")
from offline2 import env_files, groups

d, truth, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
G = np.load('scratch/zone_geo.npz'); ZD = (G['ZD'] + G['ZD'].T) / 2
W = np.exp(-ZD / 300.0); np.fill_diagonal(W, 0)
_, V = np.linalg.eigh(np.diag(W.sum(1)) - W)
BASES = {"z63": np.eye(63), "lap32": V[:, :32], "lap16": V[:, :16]}
LAMS = (0.5, 0.8, 0.9, 0.95, 0.97, 0.99)


def tf(X, M):
    return np.hstack([X[:, :3]] + [X[:, 3 + 63 * k:3 + 63 * (k + 1)] @ M for k in range(3)] + [np.ones((len(X), 1))])


def stats(X, r):
    N, D1 = X.shape
    X0, X1, y = X[:-1], X[1:], r[1:]
    return dict(Cd=X0.T @ (X0 - X1), b=X0.T @ (y - y.mean()), n=N - 1, s1=X0.sum(0), s2=(X0 ** 2).sum(0),
                sr=y.sum())


def solve(P, ridge=1e-6):
    n = P["n"]
    sd = np.sqrt(np.maximum(P["s2"] / n - (P["s1"] / n) ** 2, 0))
    sd = np.where(sd > 1e-6, sd, 1.0); sd[-1] = 1.0
    A = P["Cd"] / n / np.outer(sd, sd)
    return np.linalg.solve(A + ridge * np.eye(len(A)), P["b"] / n / sd) / sd


def add(a, b, sgn=1):
    return {k: a[k] + sgn * b[k] for k in a}


rows, envs, St = [], {}, {}
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
    imm = np.sum(s * (r[didx] - rcf[didx]))
    w = np.zeros(N); w[didx] = 2 * s          # IPW weight x 1[differ], p = .5
    naive = np.mean(w * r)
    for bn, M in BASES.items():
        XR = tf(X, M); dXR = tf(dX, M); dXR[:, -1] = 0
        dv = (s[:, None] * dXR).sum(0)
        S = stats(XR, r)
        K = 5; bnd = np.linspace(0, N, K + 1).astype(int)
        parts = [stats(XR[bnd[k]:bnd[k + 1]], r[bnd[k]:bnd[k + 1]]) for k in range(K)]
        St[(bn, ei)] = S
        rbar = S["sr"] / S["n"]
        # trace pieces: tr = sum e_t (r_t - rbar) + th . sum e_t (phi_t - phi_{t-1})
        dphi_td = np.vstack([np.zeros((1, XR.shape[1])), XR[1:] - XR[:-1]])
        tr_pieces = {}
        for lam in LAMS:
            e = lfilter([1.0], [1.0, -lam], w)
            tr_pieces[lam] = (np.sum(e[1:] * (r[1:] - rbar)), e[1:] @ dphi_td[1:])
        envs[(bn, ei)] = dict(N=N, imm=imm, dv=dv, tr=tr_pieces, naive=naive)
        def ests(th):
            o = {"paired": (imm - dv @ th) / N}
            for lam, (a, bvec) in tr_pieces.items():
                o[f"tr{lam}"] = (a + bvec @ th) / N
            return o
        th = solve(S)
        full = ests(th)
        loo = [ests(solve(add(S, parts[k], -1))) for k in range(K)]
        row = dict(f=bn, env=ei, naive=naive, **full)
        for k in full:
            row[k + "_jk"] = K * full[k] - (K - 1) * np.mean([l[k] for l in loo])
        rows.append(row)
    del X, x
    print("env", ei, flush=True)

# pooled-theta paired estimate
for bn in BASES:
    keys = [k for k in St if k[0] == bn]
    P = St[keys[0]]
    for k in keys[1:]:
        P = add(P, St[k])
    th = solve(P)
    for (_, ei) in keys:
        e = envs[(bn, ei)]
        rows.append(dict(f=bn + "|pooled", env=ei, paired=(e["imm"] - e["dv"] @ th) / e["N"],
                         **{f"tr{l}": (a + b @ th) / e["N"] for l, (a, b) in e["tr"].items()}))
df = pd.DataFrame(rows); df.to_csv(out, index=False)
pd.set_option("display.width", 250)
cols = [c for c in df.columns if c not in ("f", "env")]
res = df.groupby("f")[cols].mean().T
res_sd = df.groupby("f")[cols].std().T
print(f"truth (DQ estimand) {truth}")
print((res.round(2).astype(str) + "±" + res_sd.round(2).astype(str)).to_string())
