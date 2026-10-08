"""Plot estimator bias, RMSE, and variance over timesteps.

Usage:
    python scripts/plot_convergence.py <run_csv> <ate_csv> <output_plot>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats


def load_run(run_path):
    df = pd.read_csv(run_path)
    skip = {"env_id", "steps", "filename", "trial"}
    estimator_cols = [c for c in df.columns if c not in skip]
    return df.melt(
        id_vars=["env_id", "steps"],
        value_vars=estimator_cols,
        var_name="estimator",
        value_name="estimate",
    )


def load_true_ate(ate_path):
    ate = pd.read_csv(ate_path)
    reward = ate[ate["metric"] == "reward"]
    return (
        reward[reward["treatment"] == "B"]["value"].mean()
        - reward[reward["treatment"] == "A"]["value"].mean()
    )


def compute_metrics(run_long, true_ate):
    scale = abs(true_ate)
    z = 1.96  # 95% CI
    records = []
    for (steps, estimator), grp in run_long.groupby(["steps", "estimator"]):
        est = grp["estimate"].values
        errors = est - true_ate
        n = len(est)

        # Bias: CI via SE of the mean
        bias = np.mean(errors)
        bias_se = np.std(errors, ddof=1) / np.sqrt(n)

        # RMSE: CI via delta method on mean squared error
        sq_errors = errors**2
        mse = np.mean(sq_errors)
        mse_se = np.std(sq_errors, ddof=1) / np.sqrt(n)
        rmse = np.sqrt(mse)
        rmse_lo = np.sqrt(max(0.0, mse - z * mse_se))
        rmse_hi = np.sqrt(mse + z * mse_se)

        # Std: CI via chi-squared distribution
        std = np.std(est, ddof=1)
        if n > 1:
            chi2_lo = stats.chi2.ppf(0.975, df=n - 1)
            chi2_hi = stats.chi2.ppf(0.025, df=n - 1)
            std_lo = std * np.sqrt((n - 1) / chi2_lo)
            std_hi = std * np.sqrt((n - 1) / chi2_hi)
        else:
            std_lo = std_hi = std

        records.append(
            {
                "steps": steps,
                "estimator": estimator,
                "bias": bias / scale * 100,
                "bias_lo": (bias - z * bias_se) / scale * 100,
                "bias_hi": (bias + z * bias_se) / scale * 100,
                "rmse": rmse / scale * 100,
                "rmse_lo": rmse_lo / scale * 100,
                "rmse_hi": rmse_hi / scale * 100,
                "std": std / scale * 100,
                "std_lo": std_lo / scale * 100,
                "std_hi": std_hi / scale * 100,
            }
        )
    return pd.DataFrame(records)


def make_plot(metrics):
    estimators = metrics["estimator"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(estimators)))
    color_map = dict(zip(estimators, colors))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for ax, (metric, log_scale, ylabel) in zip(
        axes,
        [
            ("bias", False, "Bias (% |ATE|)"),
            ("rmse", True, "RMSE (% |ATE|)"),
            ("std", True, "Std Dev (% |ATE|)"),
        ],
    ):
        for estimator, grp in metrics.groupby("estimator"):
            grp = grp.sort_values("steps")
            color = color_map[estimator]
            ax.plot(
                grp["steps"],
                grp[metric],
                label=estimator,
                color=color,
                linewidth=1.5,
            )
            ax.fill_between(
                grp["steps"],
                grp[f"{metric}_lo"],
                grp[f"{metric}_hi"],
                color=color,
                alpha=0.15,
            )
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        else:
            ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Steps")
    axes[0].legend(
        title="Estimator", bbox_to_anchor=(1.01, 1), loc="upper left"
    )
    axes[0].set_title("Estimator Convergence")

    fig.tight_layout()
    return fig


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <run_csv> <ate_csv> <output_plot>")
        sys.exit(1)

    run_path, ate_path, output_plot = sys.argv[1:]

    run_long = load_run(run_path)
    true_ate = load_true_ate(ate_path)
    print(f"True ATE: {true_ate:.6f}")

    metrics = compute_metrics(run_long, true_ate)

    fig = make_plot(metrics)

    if d := os.path.dirname(output_plot):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_plot}")

    stem = os.path.splitext(output_plot)[0]
    csv_path = stem + ".csv"
    metrics.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
