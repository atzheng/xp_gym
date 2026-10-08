"""Summarize final-step estimates vs the true ATE.

usage: python scripts/summarize_lstd.py stb run_csv [run_csv ...]
ATE per threshold: B=0.1 from dq/ate/ate_stb.csv (1000 runs); others from
lambda(1)-lambda(0) of the CRN lambda(p) sweep (64 runs, SE ~0.06).
"""
import os, sys
import numpy as np, pandas as pd

SO = {"client_kwargs": {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL")}}


def true_ate(stb):
    if abs(stb - 0.1) < 1e-9:
        a = pd.read_csv("s3://research/dq/ate/ate_stb.csv", storage_options=SO)
        a = a[a.metric == "reward"].groupby("treatment").value.mean()
        return a["B"] - a["A"]
    d = pd.read_csv(f"s3://research/dq/lambda/lambda_stb{stb}.csv", storage_options=SO)
    w = d.groupby(["p", "env"]).reward.mean().unstack(0)
    return (w[1.0] - w[0.0]).mean()


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
    stb = float(sys.argv[1])
    ate = true_ate(stb)
    dfs = [pd.read_csv(f, storage_options=SO if f.startswith("s3://") else None) for f in sys.argv[2:]]
    df = dfs[0]
    for d in dfs[1:]:
        new = [c for c in d.columns if c not in df.columns]
        # same seed => identical trajectories; check via the naive column
        chk = df.merge(d[["env_id", "steps", "trial", "naive"]], on=["env_id", "steps", "trial"])
        assert np.allclose(chk.naive_x, chk.naive_y, rtol=1e-4), "runs are not paired"
        df = df.merge(d[["env_id", "steps", "trial"] + new], on=["env_id", "steps", "trial"])
    print(f"B={stb}  ATE={ate:.3f}")
    pd.set_option("display.width", 200)
    print(summarize(df, ate).round(2).to_string(index=False))
