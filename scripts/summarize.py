"""Summarize experiment results: bias, std dev, RMSE (as % of |ATE|) per estimator and setting.

Reads run and ate CSV files (glob patterns supported), computes metrics
with 95% bootstrap confidence intervals, saves a summary CSV and plot.

Usage:
    python scripts/summarize.py <run_csv> <ate_csv> <output_csv> <output_plot>

    <run_csv> and <ate_csv> may be glob patterns, e.g.:
        'outputs/pool/mat*_stb*/run.csv'
"""

import sys
import os
import duckdb
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_errorbar, geom_line, geom_hline,
    facet_grid, labs, theme_bw, theme, element_text, scale_color_brewer,
    position_dodge,
)

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05


def bootstrap_ci(arr, stat_fn, n=N_BOOTSTRAP, alpha=CI_ALPHA, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    samples = arr[idx]
    stats = np.apply_along_axis(stat_fn, 1, samples)
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return lo, hi


def compute_group_metrics(estimates, true_ate, rng):
    scale = abs(true_ate)
    errors = estimates - true_ate

    def pct(x):
        return x / scale * 100

    bias = pct(np.mean(errors))
    std = pct(np.std(errors, ddof=1))
    rmse = pct(np.sqrt(np.mean(errors ** 2)))

    bias_lo, bias_hi = bootstrap_ci(errors, lambda x: pct(np.mean(x)), rng=rng)
    std_lo, std_hi = bootstrap_ci(errors, lambda x: pct(np.std(x, ddof=1)), rng=rng)
    rmse_lo, rmse_hi = bootstrap_ci(
        errors, lambda x: pct(np.sqrt(np.mean(x ** 2))), rng=rng
    )

    return {
        "bias": bias, "bias_lo": bias_lo, "bias_hi": bias_hi,
        "std": std, "std_lo": std_lo, "std_hi": std_hi,
        "rmse": rmse, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
    }


def load_data(run_path, ate_path):
    con = duckdb.connect()

    run_raw = con.execute(f"""
        SELECT *,
            TRY_CAST(nullif(regexp_extract(filename, 'mat(\\d+)', 1), '') AS INTEGER) AS mat,
            TRY_CAST(nullif(regexp_extract(filename, 'stb([0-9.]+)', 1), '') AS DOUBLE) AS stb,
        FROM read_csv('{run_path}', filename = true)
    """).df()
    run_raw["mat"] = run_raw["mat"].fillna(0).astype(int)
    run_raw["stb"] = run_raw["stb"].fillna(0.0)
    run_raw = run_raw.drop("trial", axis=1)  

    ate_raw = con.execute(f"""
        SELECT *,
            TRY_CAST(nullif(regexp_extract(filename, 'mat(\\d+)', 1), '') AS INTEGER) AS mat,
            TRY_CAST(nullif(regexp_extract(filename, 'stb([0-9.]+)', 1), '') AS DOUBLE) AS stb,
        FROM read_csv('{ate_path}', filename = true)
    """).df()
    ate_raw["mat"] = ate_raw["mat"].fillna(0).astype(int)
    ate_raw["stb"] = ate_raw["stb"].fillna(0.0)

    con.close()
    return run_raw, ate_raw


def get_final_estimates(run_raw):
    """Keep only the last reported step per (env_id, mat, stb)."""
    max_steps = (
        run_raw.groupby(["mat", "stb", "env_id"])["steps"]
        .transform("max")
    )
    return run_raw[run_raw["steps"] == max_steps].copy()


def compute_true_ate(ate_raw):
    if "treatment" in ate_raw.columns and "metric" in ate_raw.columns:
        reward_df = ate_raw[ate_raw["metric"] == "reward"]
        return (
            reward_df.groupby(["mat", "stb"])
            .apply(lambda g:
                g[g["treatment"] == "B"]["value"].mean()
                - g[g["treatment"] == "A"]["value"].mean()
            )
            .reset_index(name="true_ate")
        )
    else:
        return (
            ate_raw.groupby(["mat", "stb"])
            .apply(lambda g: g["B"].mean() - g["A"].mean())
            .reset_index(name="true_ate")
        )


def melt_estimators(run_final):
    skip = {"env_id", "steps", "mat", "stb", "filename"}
    estimator_cols = [c for c in run_final.columns if c not in skip]
    return run_final.melt(
        id_vars=["mat", "stb", "env_id"],
        value_vars=estimator_cols,
        var_name="estimator",
        value_name="estimate",
    )


def build_summary(run_long, true_ate_df):
    rng = np.random.default_rng(42)
    merged = run_long.merge(true_ate_df, on=["mat", "stb"])

    records = []
    for (mat, stb, estimator), grp in merged.groupby(["mat", "stb", "estimator"]):
        estimates = grp["estimate"].values
        true_ate = grp["true_ate"].iloc[0]
        metrics = compute_group_metrics(estimates, true_ate, rng)
        records.append({
            "mat": mat,
            "stb": stb,
            "estimator": estimator,
            "n": len(estimates),
            "true_ate": true_ate,
            **metrics,
        })

    return pd.DataFrame(records)


def build_plot_df(summary):
    rows = []
    for _, r in summary.iterrows():
        for metric, val, lo, hi in [
            ("bias (%)",  r["bias"], r["bias_lo"], r["bias_hi"]),
            ("std (%)",   r["std"],  r["std_lo"],  r["std_hi"]),
            ("rmse (%)",  r["rmse"], r["rmse_lo"], r["rmse_hi"]),
        ]:
            rows.append({
                "mat": r["mat"],
                "stb": r["stb"],
                "estimator": r["estimator"],
                "metric": metric,
                "value": val,
                "lo": lo,
                "hi": hi,
            })
    df = pd.DataFrame(rows)
    df["stb"] = df["stb"].astype(str)
    df["mat_label"] = "max_trips=" + df["mat"].astype(str)
    return df


def make_plot(plot_df):
    dodge = position_dodge(width=0.2)

    hline_df = pd.DataFrame({"metric": ["bias (%)"], "yintercept": [0.0]})

    p = (
        ggplot(plot_df, aes(x="stb", y="value", color="estimator", group="estimator"))
        + geom_hline(
            aes(yintercept="yintercept"),
            data=hline_df,
            linetype="dashed",
            color="#808080",
            size=0.5,
        )
        + geom_line(position=dodge, size=0.7, alpha=0.8)
        + geom_point(position=dodge, size=2)
        + geom_errorbar(
            aes(ymin="lo", ymax="hi"),
            position=dodge,
            width=0.15,
            size=0.6,
        )
        + facet_grid("metric ~ mat_label", scales="free_y")
        + scale_color_brewer(type="qual", palette="Set2")
        + labs(
            x="Savings Threshold B",
            y="% of |ATE| (95% bootstrap CI)",
            color="Estimator",
            title="Estimator Performance vs True ATE",
        )
        + theme_bw()
        + theme(
            axis_text_x=element_text(size=9),
            strip_text=element_text(size=9),
            legend_position="bottom",
        )
    )
    return p


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <run_csv> <ate_csv> <output_csv> <output_plot>")
        sys.exit(1)

    run_path, ate_path, output_csv, output_plot = sys.argv[1:]

    print("Loading data...")
    run_raw, ate_raw = load_data(run_path, ate_path)

    print("Computing final estimates...")
    run_final = get_final_estimates(run_raw)
    true_ate_df = compute_true_ate(ate_raw)

    print(f"True ATEs:\n{true_ate_df.to_string(index=False)}")

    run_long = melt_estimators(run_final)

    print("Computing bias, std, RMSE (% of |ATE|) with bootstrap CIs...")
    summary = build_summary(run_long, true_ate_df)

    if d := os.path.dirname(output_csv):
        os.makedirs(d, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"Saved summary to {output_csv}")

    print(summary.to_string(index=False))

    plot_df = build_plot_df(summary)
    p = make_plot(plot_df)

    if d := os.path.dirname(output_plot):
        os.makedirs(d, exist_ok=True)
    p.save(output_plot, width=10, height=8, dpi=150, verbose=False)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()
