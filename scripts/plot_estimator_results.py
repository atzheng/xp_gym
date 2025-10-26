#!/usr/bin/env python3
"""
Plot mean estimations with confidence intervals for different estimators over time.
"""

import argparse
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def parse_ate_argument(ate_arg):
    """
    Parse the ate argument which can be either a float or a CSV file path.
    If it's a CSV file, compute ATE as mean(B) - mean(A).
    """
    try:
        # First try to parse as float
        return float(ate_arg)
    except ValueError:
        # If that fails, try to read as CSV file
        try:
            df = pd.read_csv(ate_arg)
            if "A" not in df.columns or "B" not in df.columns:
                raise ValueError(
                    f"CSV file {ate_arg} must contain columns 'A' and 'B'"
                )
            ate = df["B"].mean() - df["A"].mean()
            print(f"Computed ATE from {ate_arg}: {ate:.6f}")
            return ate
        except Exception as e:
            raise ValueError(
                f"Could not parse ate argument '{ate_arg}' as float or CSV file: {e}"
            )


def compute_summary_stats(df, true_ate, output_path=None):
    """
    Compute bias, variance, and RMSE for each estimator at the final time step.
    Optionally save to CSV if output_path is provided.
    """
    # Get final estimates for each environment
    final_estimates = df[df["steps"] == df["steps"].max()]
    estimator_cols = [
        col for col in df.columns if col not in ["env_id", "steps"]
    ]

    summary_stats = []

    for estimator in estimator_cols:
        estimates = final_estimates[estimator]

        # Compute statistics
        mean_estimate = estimates.mean()
        bias = mean_estimate - true_ate
        variance = estimates.var(ddof=1)  # Sample variance
        rmse = np.sqrt(((estimates - true_ate) ** 2).mean())

        summary_stats.append(
            {
                "estimator": estimator,
                "bias": bias,
                "variance": variance,
                "rmse": rmse,
                "mean_estimate": mean_estimate,
                "true_ate": true_ate,
                "n_samples": len(estimates),
            }
        )

    summary_df = pd.DataFrame(summary_stats)

    # Print summary
    print("\n=== Summary Statistics (Final Time Step) ===")
    print(
        f"{'Estimator':<15} {'Bias':<10} {'Variance':<10} {'RMSE':<10} {'Mean Est.':<10}"
    )
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(
            f"{row['estimator']:<15} {row['bias']:<10.6f} {row['variance']:<10.6f} {row['rmse']:<10.6f} {row['mean_estimate']:<10.6f}"
        )

    # Save to CSV if path provided
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary statistics saved to: {output_path}")

    return summary_df


def plot_estimator_results(
    csv_path,
    output_path,
    true_ate,
):
    """
    Plot mean estimations with 95% confidence intervals for different estimators.
    """
    # Read the data
    df = pd.read_csv(csv_path)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique env_ids: {sorted(df['env_id'].unique())}")
    print(f"Steps range: {df['steps'].min()} - {df['steps'].max()}")

    # Estimator columns (excluding env_id and steps)
    estimator_cols = [
        col for col in df.columns if col not in ["env_id", "steps"]
    ]
    print(f"Estimators: {estimator_cols}")

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Colors for each estimator
    colors = plt.cm.Set1(np.linspace(0, 1, len(estimator_cols)))

    # Plot each estimator
    for i, estimator in enumerate(estimator_cols):
        # Group by steps and calculate statistics across environments
        stats_by_step = (
            df.groupby("steps")[estimator]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                    lambda x: np.percentile(x, 2.5),  # Lower 95% CI
                    lambda x: np.percentile(x, 97.5),  # Upper 95% CI
                ]
            )
            .reset_index()
        )

        stats_by_step.columns = [
            "steps",
            "mean",
            "std",
            "count",
            "ci_lower",
            "ci_upper",
        ]

        # Calculate standard error and confidence intervals using t-distribution
        stats_by_step["se"] = stats_by_step["std"] / np.sqrt(
            stats_by_step["count"]
        )
        stats_by_step["ci_t_lower"] = (
            stats_by_step["mean"]
            - stats.t.ppf(0.975, stats_by_step["count"] - 1)
            * stats_by_step["se"]
        )
        stats_by_step["ci_t_upper"] = (
            stats_by_step["mean"]
            + stats.t.ppf(0.975, stats_by_step["count"] - 1)
            * stats_by_step["se"]
        )

        # Use t-distribution CIs where we have enough samples, otherwise use percentile CIs
        use_t_ci = stats_by_step["count"] >= 3
        final_lower = np.where(
            use_t_ci, stats_by_step["ci_t_lower"], stats_by_step["ci_lower"]
        )
        final_upper = np.where(
            use_t_ci, stats_by_step["ci_t_upper"], stats_by_step["ci_upper"]
        )

        # Plot the line and confidence interval
        plt.plot(
            stats_by_step["steps"],
            stats_by_step["mean"],
            color=colors[i],
            label=estimator,
            linewidth=2,
            alpha=0.9,
        )
        plt.fill_between(
            stats_by_step["steps"],
            final_lower,
            final_upper,
            color=colors[i],
            alpha=0.2,
        )

    # Add true ATE horizontal line
    plt.axhline(
        y=true_ate,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"True ATE = {true_ate:.3f}",
        alpha=0.8,
    )

    plt.xlabel("Time Steps (n_events)", fontsize=12)
    plt.ylabel("Treatment Effect Estimate", fontsize=12)
    plt.title(
        "Estimator Convergence: Mean ± 95% Confidence Interval",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Also show summary statistics
    print("\n=== Summary Statistics (Final Time Step) ===")
    final_step_data = df[df["steps"] == df["steps"].max()]

    for estimator in estimator_cols:
        values = final_step_data[estimator]
        print(
            f"{estimator:15s}: mean={values.mean():8.4f}, std={values.std():8.4f}, "
            f"range=[{values.min():8.4f}, {values.max():8.4f}]"
        )

    # Show convergence analysis
    print("\n=== Convergence Analysis (Last 10% of steps) ===")
    last_10pct_steps = df["steps"] >= df["steps"].quantile(0.9)
    last_data = df[last_10pct_steps]

    for estimator in estimator_cols:
        values = last_data[estimator]
        trend_slope = np.polyfit(last_data["steps"], values, 1)[0]
        print(
            f"{estimator:15s}: trend_slope={trend_slope:10.6f} (closer to 0 = more converged)"
        )

    # plt.show()  # Comment out to avoid hanging
    return df


def plot_estimator_comparison(
    csv_path,
    output_path,
    true_ate,
):
    """
    Create a comparison plot showing the distribution of final estimates.
    """
    df = pd.read_csv(csv_path)

    # Get final estimates for each environment
    final_estimates = df[df["steps"] == df["steps"].max()]
    estimator_cols = [
        col for col in df.columns if col not in ["env_id", "steps"]
    ]

    # Melt the data for easier plotting
    melted = final_estimates[["env_id"] + estimator_cols].melt(
        id_vars=["env_id"],
        value_vars=estimator_cols,
        var_name="Estimator",
        value_name="Final_Estimate",
    )

    # Create box plot
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(data=melted, x="Estimator", y="Final_Estimate")
    # Control y-axis limits around the true ATE

    sns.stripplot(
        data=melted,
        x="Estimator",
        y="Final_Estimate",
        color="red",
        alpha=0.6,
        size=4,
    )

    # Add mean points
    means = melted.groupby("Estimator")["Final_Estimate"].mean()
    ymin, ymax = true_ate - 3, true_ate + 3
    ax.set_ylim(ymin, ymax)

    # When plotting means, clip them inside the ylim
    for i, estimator in enumerate(estimator_cols):
        mean_val = means[estimator]
        # Clip to visible range
        clipped_val = np.clip(mean_val, ymin, ymax)
        ax.scatter(
            i,
            clipped_val,
            color="blue",
            s=100,
            marker="D",
            label="Mean" if i == 0 else "",
            zorder=5,
            edgecolors="white",
            linewidth=1,
        )
        # Add mean value text (also clipped)
        ax.text(
            i,
            clipped_val + 0.05,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add true ATE horizontal line
    ax.axhline(
        y=true_ate,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"True ATE = {true_ate:.3f}",
        alpha=0.8,
    )

    ax.set_xticklabels(estimator_cols, rotation=0, ha="center")

    # plt.title('Distribution of Final Treatment Effect Estimates', fontsize=14, fontweight='bold')
    plt.ylabel("Estimated ATE", fontsize=12)
    plt.xlabel("Estimator", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Print bias analysis
    print(f"\n=== Bias Analysis (True ATE = {true_ate:.6f}) ===")
    for estimator in estimator_cols:
        mean_estimate = means[estimator]
        bias = mean_estimate - true_ate
        abs_bias = abs(bias)
        print(
            f"{estimator:15s}: bias={bias:8.4f}, |bias|={abs_bias:8.4f}, mean={mean_estimate:8.4f}"
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {output_path}")
    # plt.show()  # Comment out to avoid hanging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mean estimations with confidence intervals for different estimators over time."
    )
    parser.add_argument(
        "csv_path", help="Path to the CSV file containing estimator results"
    )
    parser.add_argument(
        "--ate",
        type=str,
        required=True,
        help="True Average Treatment Effect value (float) or path to CSV file from compute-ate.py",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Output directory for all generated files (default: plots)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Define output paths using the output directory
    convergence_output = os.path.join(
        args.output_dir, "estimator_convergence.png"
    )
    comparison_output = os.path.join(
        args.output_dir, "estimator_comparison.png"
    )
    summary_stats_output = os.path.join(
        args.output_dir, "estimator_summary_stats.csv"
    )

    # Parse the ate argument (could be float or CSV path)
    true_ate = parse_ate_argument(args.ate)

    # Plot convergence over time
    df = plot_estimator_results(
        csv_path=args.csv_path,
        output_path=convergence_output,
        true_ate=true_ate,
    )

    # Plot final estimate comparison
    plot_estimator_comparison(
        csv_path=args.csv_path, output_path=comparison_output, true_ate=true_ate
    )

    # Compute and save summary statistics
    summary_stats = compute_summary_stats(
        df=df, true_ate=true_ate, output_path=summary_stats_output
    )
