"""
Analyze and visualize sparse attention benchmark results.

Reads CSV outputs from run_comparison.py and ablation_study.py, generates:
- Accuracy vs sparsity Pareto curves
- Latency vs prompt length plots
- Ablation heatmaps
- Statistical significance tests (paired bootstrap CI)

Usage:
  python benchmark/sparse_attention/analyze_results.py \
    --comparison-csv results/sparse_comparison/comparison_*.csv \
    --ablation-csv results/ablation/ablation_*.csv \
    --output-dir results/figures/
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sparse attention results")
    parser.add_argument("--comparison-csv", type=str, nargs="+", default=[])
    parser.add_argument("--ablation-csv", type=str, nargs="+", default=[])
    parser.add_argument("--output-dir", type=str, default="results/figures/")
    parser.add_argument("--bootstrap-n", type=int, default=10000, help="Bootstrap resamples")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    return parser.parse_args()


def load_csv(paths):
    """Load and merge CSV files."""
    rows = []
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows.extend(list(reader))
    return rows


def paired_bootstrap_ci(scores_a, scores_b, n_resamples=10000, alpha=0.05):
    """Paired bootstrap confidence interval for difference in means.

    Returns (mean_diff, ci_lower, ci_upper, p_value).
    """
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    n = len(scores_a)
    assert n == len(scores_b), "Score arrays must have equal length"

    observed_diff = scores_a.mean() - scores_b.mean()

    # Bootstrap resampling
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_diff = scores_a[idx].mean() - scores_b[idx].mean()
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    # Two-sided p-value (proportion of bootstraps on wrong side of 0)
    if observed_diff >= 0:
        p_value = (boot_diffs <= 0).mean() * 2
    else:
        p_value = (boot_diffs >= 0).mean() * 2
    p_value = min(p_value, 1.0)

    return observed_diff, ci_lower, ci_upper, p_value


def holm_bonferroni_correction(p_values, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons.

    Returns list of (index, p_value, adjusted_alpha, significant) tuples.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = []
    for rank, (idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        significant = p <= adjusted_alpha
        results.append((idx, p, adjusted_alpha, significant))
        if not significant:
            # Mark all remaining as non-significant
            for remaining_rank in range(rank + 1, m):
                r_idx, r_p = indexed[remaining_rank]
                r_alpha = alpha / (m - remaining_rank)
                results.append((r_idx, r_p, r_alpha, False))
            break
    results.sort(key=lambda x: x[0])
    return results


def analyze_comparison(rows, output_dir, bootstrap_n, alpha):
    """Analyze comparison results."""
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)

    # Group by (algorithm, sparsity_ratio, task)
    grouped = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], float(row["sparsity_ratio"]), row["task"])
        grouped[key].append(float(row["accuracy"]))

    # Print summary table
    print(f"\n{'Algorithm':<12} {'Sparsity':<10} {'Task':<10} {'Accuracy':<10} {'N':<5}")
    print("-" * 52)
    for key in sorted(grouped.keys()):
        algo, ratio, task = key
        scores = grouped[key]
        mean_acc = np.mean(scores)
        print(f"{algo:<12} {ratio:<10.1f} {task:<10} {mean_acc:<10.4f} {len(scores):<5}")

    # Statistical tests: compare quest vs h2oquest at each sparsity ratio
    print("\n\nStatistical Comparisons (Quest vs H2OQuest):")
    print("-" * 70)
    p_values = []
    comparisons = []

    for task in set(row["task"] for row in rows):
        for ratio in sorted(set(float(row["sparsity_ratio"]) for row in rows if row["algorithm"] != "dense")):
            quest_key = ("quest", ratio, task)
            h2o_key = ("h2oquest", ratio, task)

            if quest_key in grouped and h2o_key in grouped:
                quest_scores = grouped[quest_key]
                h2o_scores = grouped[h2o_key]

                if len(quest_scores) > 1 and len(h2o_scores) > 1:
                    n = min(len(quest_scores), len(h2o_scores))
                    diff, ci_lo, ci_hi, p = paired_bootstrap_ci(
                        h2o_scores[:n], quest_scores[:n], bootstrap_n, alpha
                    )
                    p_values.append(p)
                    comparisons.append((task, ratio, diff, ci_lo, ci_hi, p))

    if p_values:
        corrections = holm_bonferroni_correction(p_values, alpha)
        for i, (task, ratio, diff, ci_lo, ci_hi, p) in enumerate(comparisons):
            _, _, adj_alpha, sig = corrections[i]
            sig_str = "*" if sig else " "
            print(
                f"  {task:<8} ratio={ratio:.1f}: "
                f"H2OQuest-Quest = {diff:+.4f} "
                f"[{ci_lo:+.4f}, {ci_hi:+.4f}] "
                f"p={p:.4f} {sig_str}"
            )

    # Generate accuracy vs sparsity data for plotting
    pareto_path = os.path.join(output_dir, "accuracy_vs_sparsity.csv")
    with open(pareto_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "sparsity_ratio", "task", "mean_accuracy", "std_accuracy", "n"])
        for key in sorted(grouped.keys()):
            algo, ratio, task = key
            scores = grouped[key]
            writer.writerow([algo, ratio, task, np.mean(scores), np.std(scores), len(scores)])
    print(f"\nPareto data saved to: {pareto_path}")

    return grouped


def analyze_ablation(rows, output_dir):
    """Analyze ablation results."""
    print("\n" + "=" * 60)
    print("ABLATION ANALYSIS")
    print("=" * 60)

    # Group by sweep_param
    by_param = defaultdict(list)
    for row in rows:
        by_param[row["sweep_param"]].append(row)

    for param, param_rows in sorted(by_param.items()):
        print(f"\n  {param}:")
        print(f"  {'Value':<15} {'Accuracy':<10}")
        print(f"  {'-'*30}")

        values_scores = []
        for row in sorted(param_rows, key=lambda r: float(r["sweep_value"])):
            value = row["sweep_value"]
            accuracy = float(row["accuracy"])
            print(f"  {value:<15} {accuracy:<10.4f}")
            values_scores.append((float(value), accuracy))

        # Find best value
        if values_scores:
            best_val, best_acc = max(values_scores, key=lambda x: x[1])
            print(f"  Best: {param}={best_val} (accuracy={best_acc:.4f})")

    # Save ablation summary
    ablation_path = os.path.join(output_dir, "ablation_summary.csv")
    with open(ablation_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep_param", "sweep_value", "accuracy"])
        for row in rows:
            writer.writerow([row["sweep_param"], row["sweep_value"], row["accuracy"]])
    print(f"\nAblation summary saved to: {ablation_path}")


def generate_plots(output_dir):
    """Generate matplotlib plots if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    # Plot accuracy vs sparsity
    pareto_path = os.path.join(output_dir, "accuracy_vs_sparsity.csv")
    if os.path.exists(pareto_path):
        data = defaultdict(lambda: defaultdict(list))
        with open(pareto_path) as f:
            for row in csv.DictReader(f):
                algo = row["algorithm"]
                ratio = float(row["sparsity_ratio"])
                acc = float(row["mean_accuracy"])
                data[algo][ratio].append(acc)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        colors = {"dense": "black", "quest": "blue", "h2oquest": "red"}
        markers = {"dense": "s", "quest": "o", "h2oquest": "^"}

        for algo in ["dense", "quest", "h2oquest"]:
            if algo in data:
                ratios = sorted(data[algo].keys())
                accs = [np.mean(data[algo][r]) for r in ratios]
                ax.plot(
                    ratios, accs,
                    color=colors.get(algo, "gray"),
                    marker=markers.get(algo, "."),
                    label=algo,
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Sparsity Ratio (fraction of pages retained)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy vs Sparsity Ratio", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig_path = os.path.join(output_dir, "accuracy_vs_sparsity.pdf")
        fig.savefig(fig_path, dpi=300)
        print(f"Plot saved: {fig_path}")
        plt.close(fig)

    # Plot ablation heatmap
    ablation_path = os.path.join(output_dir, "ablation_summary.csv")
    if os.path.exists(ablation_path):
        by_param = defaultdict(list)
        with open(ablation_path) as f:
            for row in csv.DictReader(f):
                by_param[row["sweep_param"]].append(
                    (float(row["sweep_value"]), float(row["accuracy"]))
                )

        n_params = len(by_param)
        if n_params > 0:
            fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
            if n_params == 1:
                axes = [axes]

            for ax, (param, vals) in zip(axes, sorted(by_param.items())):
                vals.sort()
                x = [v[0] for v in vals]
                y = [v[1] for v in vals]
                ax.bar(range(len(x)), y, color="steelblue", alpha=0.8)
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels([str(v) for v in x], rotation=45)
                ax.set_xlabel(param)
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Ablation: {param}")

            fig.tight_layout()
            fig_path = os.path.join(output_dir, "ablation_heatmap.pdf")
            fig.savefig(fig_path, dpi=300)
            print(f"Plot saved: {fig_path}")
            plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.comparison_csv:
        comparison_rows = load_csv(args.comparison_csv)
        if comparison_rows:
            analyze_comparison(comparison_rows, args.output_dir, args.bootstrap_n, args.alpha)
        else:
            print("No comparison data found.")

    if args.ablation_csv:
        ablation_rows = load_csv(args.ablation_csv)
        if ablation_rows:
            analyze_ablation(ablation_rows, args.output_dir)
        else:
            print("No ablation data found.")

    generate_plots(args.output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
