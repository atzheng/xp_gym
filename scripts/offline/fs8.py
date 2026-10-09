"""Bias of pure LSTD-DQ vs amount of data (theta fitted on 1..16 pooled half-envs),
for several feature sets, with and without Rao-Blackwellising the next state over z
(next features/reward averaged over both arms' outcomes, which are known).
usage: python scratch/fs5.py <dev dir> <out.csv> [set,set,...]
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scratch")
from offline2 import env_files, groups

d, out = sys.argv[1], sys.argv[2]
G = np.load('scratch/zone_geo.npz'); ZD = (G['ZD'] + G['ZD'].T) / 2; do, dd = G['do'], G['dd']
W = np.exp(-ZD / 300.0); np.fill_diagonal(W, 0)
_, V = np.linalg.eigh(np.diag(W.sum(1)) - W)   # smoothest first
def dk(ls, dem):
    P = np.stack([np.exp(-ZD / l) @ dem for l in ls], 1)
    return P / P.std(0)
DK = np.hstack([dk([120, 300, 600, 1200], do), dk([120, 300, 600, 1200], dd)])
BASES = {"z63": np.eye(63), "lap16": V[:, :16], "lap32": V[:, :32], "dk+lap8": np.hstack([V[:, :8], DK])}
BASES = {"z63": np.eye(63), "lap32": V[:, :32]}
VAR = {"glob": 0, "b100k": 100000, "b20k": 20000, "b5k": 5000, "b1k": 1000, "b200": 200}
SETS = {}
for k, M in BASES.items():
    for ns, sfx in [(0, ""), (3, "+scal")]:
        for v, L in VAR.items():
            SETS[k + sfx + "|" + v] = (M, ns, L)
if len(sys.argv) > 3:
    SETS = {k: v for k, v in SETS.items() if k.split("|")[0] in sys.argv[3].split(",")}


def tf(X, M, ns):
    return np.hstack([X[:, :3]] + [X[:, 3 + 63 * k:3 + 63 * (k + 1)] @ M for k in range(3)]
                     + [X[:, 192:192 + ns], np.ones((len(X), 1))])


from scipy.signal import lfilter
def stats(X, r, L, off=0):
    """LSTD stats with a block-constant average reward (blocks of L steps of absolute
    step index; L=0: one global rbar). Within-block demeaned (FWL) cross-products."""
    N, D1 = X.shape
    Cd = np.zeros((D1, D1)); b = np.zeros(D1); s1 = np.zeros(D1); s2 = np.zeros(D1)
    rr = r[1:]
    idx = np.arange(N - 1) + off
    bl = idx // L if L else np.zeros(N - 1, int)
    edges = np.flatnonzero(np.diff(bl)) + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [N - 1]])
    # process in groups of blocks to bound memory
    i = 0
    while i < len(starts):
        k = i
        while k < len(starts) and ends[k] - starts[i] < 50000: k += 1
        k = max(k, i + 1)
        a, e = starts[i], ends[k - 1]
        X0 = X[a:e]; dX = X0 - X[a + 1:e + 1]; y = rr[a:e]
        blk = bl[a:e] - bl[a]
        nb = blk.max() + 1
        cnt = np.bincount(blk, minlength=nb).astype(float)[:, None]
        mX0 = np.zeros((nb, D1)); np.add.at(mX0, blk, X0); mX0 /= cnt
        mdX = np.zeros((nb, D1)); np.add.at(mdX, blk, dX); mdX /= cnt
        my = np.bincount(blk, weights=y, minlength=nb) / cnt[:, 0]
        Z = X0 - mX0[blk]
        Cd += Z.T @ (dX - mdX[blk]); b += Z.T @ (y - my[blk])
        s1 += X0.sum(0); s2 += (X0 ** 2).sum(0)
        i = k
    n = N - 1
    return dict(Cd=Cd, b=b, n=n, s1=s1, s2=s2)


def solve(P, ridge=1e-6):
    n = P["n"]
    sd = np.sqrt(np.maximum(P["s2"] / n - (P["s1"] / n) ** 2, 0))
    sd = np.where(sd > 1e-6, sd, 1.0); sd[-1] = 1.0
    A = P["Cd"] / n / np.outer(sd, sd)
    return np.linalg.solve(A + ridge * np.eye(len(A)), P["b"] / n / sd) / sd


envs, St = {}, {}
for ei, f in enumerate(env_files(d)):
    x = np.load(f)
    g = groups(x["names"]); names = list(map(str, x["names"]))
    cols = g["ntrips"] + g["idle"] + g["one_final"] + g["full_drop"] + \
        [names.index(n) for n in ["busy_sum", "firstfree_sum", "busy_sq_sum"]]
    r = x["r"].astype(np.float64); N = len(r) - 1; r = r[:N]
    X = x["phi"][:N][:, cols].astype(np.float64)
    didx = x["didx"]; keep = didx < N
    dX = x["dphi"][keep][:, cols].astype(np.float64); didx = didx[keep]
    z = x["z"][:N].astype(np.float64); rcf = x["r_cf"][:N].astype(np.float64)
    s = 2 * z[didx] - 1
    # z-averaged reward (p = 0.5)
    rm = r.copy(); rm[didx] += 0.5 * (rcf[didx] - r[didx])
    envs[ei] = dict(N=N, imm=np.sum(s * (r[didx] - rcf[didx])))
    cache = {}
    for name, (M, ns, gl) in SETS.items():
        key = (id(M), ns)
        if key not in cache:
            cache = {key: (tf(X, M, ns), tf(dX, M, ns))}
        XR, dXR = cache[key]; dXR = dXR.copy(); dXR[:, -1] = 0
        envs[ei][name] = (s[:, None] * dXR).sum(0)
        h = N // 2
        St[(name, ei)] = stats(XR, r, gl)
        St[(name, ei, 0)] = stats(XR[:h], r[:h], gl)
        St[(name, ei, 1)] = stats(XR[h:], r[h:], gl, off=h)
    del X, x
    print("env", ei, flush=True)

E = len(envs)
rows = []
def pool(keys):
    return solve({k: sum(St[kk][k] for kk in keys) for k in ["Cd", "b", "n", "s1", "s2"]})
for name in SETS:
    ev = lambda th: [(e["imm"] - e[name] @ th) / e["N"] for e in envs.values()]
    for h in range(2):
        for ei in range(E):
            rows.append(dict(f=name, n_half_envs=1, est=np.mean(ev(pool([(name, ei, h)])))))
    for m in [1, 2, 4, 8]:
        for start in range(0, E, m):
            th = pool([(name, k) for k in range(start, start + m)])
            if m == 1:   # own-theta estimate for env `start` (what the online estimator does)
                rows.append(dict(f=name, n_half_envs=-2, est=ev(th)[start]))
            rows.append(dict(f=name, n_half_envs=2 * m, est=np.mean(ev(th))))
df = pd.DataFrame(rows); df.to_csv(out, index=False)
pd.set_option("display.width", 250)
t = df.groupby(["f", "n_half_envs"]).est.mean().unstack(1).round(2)
t["own_sd"] = df[df.n_half_envs == -2].groupby("f").est.std().round(2)
print(t.to_string())
