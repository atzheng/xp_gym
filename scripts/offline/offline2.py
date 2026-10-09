"""Memory-lean offline DQ-LSTD evaluation (float32 storage, float64 accumulation)."""
import glob, numpy as np
from scipy.signal import lfilter

def env_files(d): return sorted(glob.glob(f"{d}/env*.npz"))

def load_env(f, cols=None):
    x = np.load(f)
    e = {k: x[k] for k in ["z", "r", "differ", "r_cf", "didx"]}
    N = len(e["r"]) - 1  # drop final step (gymnax auto-reset)
    for k in ["z", "r", "differ", "r_cf"]: e[k] = e[k][:N]
    keep = e["didx"] < N
    e["didx"] = e["didx"][keep]
    phi = x["phi"]; dphi = x["dphi"]
    e["phi"] = (phi if cols is None else phi[:, cols])[:N].astype(np.float32)
    e["dphi"] = (dphi if cols is None else dphi[:, cols])[keep].astype(np.float32)
    e["names"] = x["names"]
    return e

def groups(names):
    g = {}
    for i, n in enumerate(map(str, names)):
        if n.startswith("ntrips"): k = "ntrips"
        elif n.startswith(("busy1", "busy2", "firstfree2")): k = n.split("_")[0]
        elif n in ("busy_sum", "busy_sq_sum", "firstfree_sum"): k = "scal"
        elif n.startswith(("idle", "one_final", "full_drop")): k = n.rsplit("_", 1)[0]
        elif n.startswith("drop_"): k = "drop"
        elif n.startswith("o"): k = n.rsplit("_k", 1)[0]  # o{off}_{rew|npool}_h{h}
        else: k = n
        g.setdefault(k, []).append(i)
    return g

def lstd_stats(X, r, chunk=50000):
    """X: (N, D) float32 (no intercept). Returns stats for LSTD of post-decision
    value: transitions (X_t -> X_{t+1}) with reward r_{t+1}."""
    N, D = X.shape
    D1 = D + 1
    C = np.zeros((D1, D1)); Cd = np.zeros((D1, D1)); b = np.zeros(D1); s1 = np.zeros(D1); s2 = np.zeros(D1)
    r1 = r[1:].astype(np.float64); rbar = r1.mean()
    for i in range(0, N - 1, chunk):
        j = min(i + chunk, N - 1)
        X0 = np.hstack([X[i:j], np.ones((j - i, 1), np.float32)]).astype(np.float64)
        X1 = np.hstack([X[i + 1:j + 1], np.ones((j - i, 1), np.float32)]).astype(np.float64)
        C += X0.T @ X1           # for (1-gamma) part
        Cd += X0.T @ (X0 - X1)   # exact differences
        b += X0.T @ (r1[i:j] - rbar)
        s1 += X0.sum(0); s2 += (X0 ** 2).sum(0)
    n = N - 1
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    return dict(C=C, Cd=Cd, b=b, n=n, rbar=rbar, sd=sd)

def lstd_solve(S, gamma, lam):
    sd = np.where(S["sd"] > 1e-6, S["sd"], 1.0); sd[-1] = 1.0
    A = (S["Cd"] + (1 - gamma) * S["C"]) / S["n"]
    A = A / np.outer(sd, sd)
    th = np.linalg.solve(A + lam * np.eye(len(A)), S["b"] / S["n"] / sd)
    return th / sd

def estimates(e, theta, gamma, rbar, lams=(0.0, 0.5, 0.8, 0.9, 0.95, 0.98)):
    X = e["phi"]
    U = X @ theta[:-1].astype(np.float32) + theta[-1]
    U = U.astype(np.float64)
    z = e["z"].astype(np.float64); s = 2 * z - 1
    r, rcf, d = e["r"].astype(np.float64), e["r_cf"].astype(np.float64), e["differ"]
    N = len(r)
    i = e["didx"]
    dU = (e["dphi"] @ theta[:-1].astype(np.float32)).astype(np.float64)
    out = {"naive": np.mean(2 * s * r * d),
           "paired": (np.sum(s[i] * (r[i] - rcf[i])) - gamma * np.sum(s[i] * dU)) / N}
    Uprev = np.concatenate([[U[0]], U[:-1]])
    delta = r - rbar + gamma * U - Uprev
    w = 2 * s * d
    for lam in lams:
        tr = lfilter([1.0], [1.0, -lam * gamma], w) if lam > 0 else w
        out[f"tr{lam}"] = np.mean(tr * delta)
    return out


def lstdl_stats(X, r, lam_v, chunk=50000):
    """LSTD(lam_v) statistics with feature eligibility traces z_t = lam_v z_{t-1} + phi_t.
    A(gamma) = Cd + (1-gamma) C with Cd = sum z_t (phi_t - phi_{t+1})^T, C = sum z_t phi_{t+1}^T."""
    N, D = X.shape
    D1 = D + 1
    C = np.zeros((D1, D1)); Cd = np.zeros((D1, D1)); b = np.zeros(D1); s1 = np.zeros(D1); s2 = np.zeros(D1)
    r1 = r[1:].astype(np.float64); rbar = r1.mean()
    zi = np.zeros((1, D1))
    for i in range(0, N - 1, chunk):
        j = min(i + chunk, N - 1)
        X0 = np.hstack([X[i:j], np.ones((j - i, 1), np.float32)]).astype(np.float64)
        X1 = np.hstack([X[i + 1:j + 1], np.ones((j - i, 1), np.float32)]).astype(np.float64)
        Z, zf = lfilter([1.0], [1.0, -lam_v], X0, axis=0, zi=zi * lam_v)
        zi = Z[-1:]
        C += Z.T @ X1; Cd += Z.T @ (X0 - X1); b += Z.T @ (r1[i:j] - rbar)
        s1 += X0.sum(0); s2 += (X0 ** 2).sum(0)
    n = N - 1
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    return dict(C=C, Cd=Cd, b=b, n=n, rbar=rbar, sd=sd)
