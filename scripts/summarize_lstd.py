"""Summarize final-step estimates vs the true ATE.

usage: python scripts/summarize_lstd.py stb run_csv [run_csv ...]
stb may be a threshold pair "A:B" (truth from ate2/ate_pair{A:B} or lambda/lambda_pair{A:B}).
ATE per threshold: B=0.1 from dq/ate/ate_stb.csv (1000 runs); others from
lambda(1)-lambda(0) of the CRN lambda(p) sweep: dq/ate2 (256 runs) if present,
else dq/lambda (64 runs, SE ~0.06).  Also prints DQ's own estimand lambda'(0.5),
estimated as (lambda(0.7)-lambda(0.3))/0.4.
"""
import os, sys
import numpy as np, pandas as pd

SO = {"client_kwargs": {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL")}}


def true_ate(stb):
    if stb == 0.1:
        a = pd.read_csv("s3://research/dq/ate/ate_stb.csv", storage_options=SO)
        a = a[a.metric == "reward"].groupby("treatment").value.mean()
        return a["B"] - a["A"]
    return _sweep(stb)[1.0] - _sweep(stb)[0.0]


def dq_estimand(stb):
    w = _sweep(stb)
    return (w[0.7] - w[0.3]) / 0.4 if 0.3 in w and 0.7 in w else np.nan


def _sweep(stb):
    kind = "pair" if ":" in str(stb) else "stb"
    for f in (f"s3://research/dq/ate2/ate_{kind}{stb}.csv", f"s3://research/dq/lambda/lambda_{kind}{stb}.csv"):
        try:
            d = pd.read_csv(f, storage_options=SO)
        except FileNotFoundError:
            continue
        if not (d.p == 0.0).any():  # p=0 doesn't depend on B; stored once per env count
            p0 = pd.read_csv(f"s3://research/dq/ate2/ate_p0_E{d.env.nunique()}.csv", storage_options=SO)
            d = pd.concat([d, p0])
        return d.groupby(["p", "env"]).reward.mean().unstack(0).mean()
    raise FileNotFoundError(stb)


def summarize(df, ate, skip=("env_id", "steps", "trial", "naive_ipw")):
    last = df[df.steps == df.steps.max()]
    rows = []
    for c in [c for c in df.columns if c not in skip]:
        x = last[c].astype(float)
        m, s, se = x.mean(), x.std(), x.std() / np.sqrt(len(x))
        bias = lambda v: (v - ate) / abs(ate) * 100
        rows.append(dict(estimator=c, n=len(x), mean=m, sd=s, bias_pct=bias(m),
                         bias_ci=f"[{bias(m - 1.96 * se):.0f}, {bias(m + 1.96 * se):.0f}]",
                         rmse=np.sqrt((m - ate) ** 2 + s ** 2), sign_ok=np.mean(np.sign(x) == np.sign(ate))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    stb = sys.argv[1] if ":" in sys.argv[1] else float(sys.argv[1])
    ate = true_ate(stb)
    dfs = [pd.read_csv(f, storage_options=SO if f.startswith("s3://") else None) for f in sys.argv[2:]]
    df = dfs[0]
    for d in dfs[1:]:
        new = [c for c in d.columns if c not in df.columns]
        # same seed => identical trajectories; check via the naive column
        chk = df.merge(d[["env_id", "steps", "trial", "naive"]], on=["env_id", "steps", "trial"])
        assert np.allclose(chk.naive_x, chk.naive_y, rtol=1e-4), "runs are not paired"
        df = df.merge(d[["env_id", "steps", "trial"] + new], on=["env_id", "steps", "trial"])
    print(f"B={stb}  ATE={ate:.3f}  DQ estimand lambda'(0.5)={dq_estimand(stb):.3f}")
    pd.set_option("display.width", 200)
    print(summarize(df, ate).round(2).to_string(index=False))
