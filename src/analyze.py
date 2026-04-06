"""
Analyze experimental results: accuracy curves, robustness gaps,
uncertainty calibration, and statistical tests.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("/workspaces/reasoning-trace-robustnes-6b65-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Budget ordering
BUDGET_ORDER = ["no_cot", "short", "medium", "long", "unconstrained"]
BUDGET_LABELS = {
    "no_cot": "No CoT",
    "short": "Short\n(≤300)",
    "medium": "Medium\n(≤800)",
    "long": "Long\n(≤1500)",
    "unconstrained": "Unconstrained\n(≤3000)",
}
BUDGET_TOKENS = {"no_cot": 150, "short": 300, "medium": 800, "long": 1500, "unconstrained": 3000}

# Color scheme
DATASET_COLORS = {
    "MATH": "#e74c3c",
    "GSM8K": "#2ecc71",
    "MMLU-Pro": "#3498db",
    "MuSR": "#9b59b6",
}

def load_results():
    """Load and prepare results DataFrame."""
    data = json.load(open(RESULTS_DIR / "raw_results.json"))
    df = pd.DataFrame(data)
    df["budget_idx"] = df["budget"].map({b: i for i, b in enumerate(BUDGET_ORDER)})
    df["budget_label"] = df["budget"].map(BUDGET_LABELS)
    df["max_tokens"] = df["budget"].map(BUDGET_TOKENS)
    return df


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for mean."""
    rng = np.random.RandomState(seed)
    means = []
    arr = np.array(values, dtype=float)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(arr), lower, upper


def compute_accuracy_table(df):
    """Compute accuracy with CIs for each (dataset, budget) pair."""
    rows = []
    for dataset in df["dataset"].unique():
        for budget in BUDGET_ORDER:
            mask = (df["dataset"] == dataset) & (df["budget"] == budget)
            correct = df.loc[mask, "correct"].values
            if len(correct) == 0:
                continue
            mean, ci_lo, ci_hi = bootstrap_ci(correct)
            avg_tokens = df.loc[mask, "completion_tokens"].mean()
            avg_conf = df.loc[mask, "confidence"].mean()
            rows.append({
                "dataset": dataset,
                "budget": budget,
                "accuracy": mean,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n": len(correct),
                "avg_completion_tokens": avg_tokens,
                "avg_confidence": avg_conf,
                "token_efficiency": mean / max(avg_tokens, 1) * 100,  # acc per 100 tokens
            })
    return pd.DataFrame(rows)


def plot_accuracy_curves(acc_df):
    """Plot accuracy vs budget for each dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in acc_df["dataset"].unique():
        sub = acc_df[acc_df["dataset"] == dataset].sort_values("budget", key=lambda x: x.map({b: i for i, b in enumerate(BUDGET_ORDER)}))
        x = np.arange(len(sub))
        ax.errorbar(x, sub["accuracy"],
                     yerr=[sub["accuracy"] - sub["ci_lower"], sub["ci_upper"] - sub["accuracy"]],
                     marker='o', capsize=4, linewidth=2, markersize=8,
                     label=dataset, color=DATASET_COLORS.get(dataset, None))

    ax.set_xticks(range(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Reasoning Budget", fontsize=12)
    ax.set_title("Accuracy vs Reasoning Trace Length (GPT-4.1)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_curves.png", dpi=150)
    plt.close()
    print("Saved: accuracy_curves.png")


def plot_actual_token_vs_accuracy(df):
    """Plot accuracy vs actual completion tokens (not budget)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        # Group by budget and compute means
        grouped = sub.groupby("budget").agg(
            acc=("correct", "mean"),
            tokens=("completion_tokens", "mean")
        ).reindex(BUDGET_ORDER)
        ax.plot(grouped["tokens"], grouped["acc"], marker='o', linewidth=2, markersize=8,
                label=dataset, color=DATASET_COLORS.get(dataset, None))

    ax.set_xlabel("Mean Completion Tokens (actual)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Actual Token Usage", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_vs_tokens.png", dpi=150)
    plt.close()
    print("Saved: accuracy_vs_tokens.png")


def plot_robustness_gap(acc_df):
    """Plot robustness gap (ID - OOD accuracy) by budget."""
    # Use MATH as ID, others as OOD
    math_acc = acc_df[acc_df["dataset"] == "MATH"].set_index("budget")["accuracy"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in ["GSM8K", "MMLU-Pro", "MuSR"]:
        ood_acc = acc_df[acc_df["dataset"] == dataset].set_index("budget")["accuracy"]
        gap = math_acc - ood_acc
        gap = gap.reindex(BUDGET_ORDER)
        x = np.arange(len(BUDGET_ORDER))
        ax.plot(x, gap.values, marker='s', linewidth=2, markersize=8,
                label=f"MATH - {dataset}", color=DATASET_COLORS.get(dataset, None))

    ax.set_xticks(range(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=10)
    ax.set_ylabel("Robustness Gap (ID acc - OOD acc)", fontsize=12)
    ax.set_xlabel("Reasoning Budget", fontsize=12)
    ax.set_title("Robustness Gap vs Reasoning Trace Length", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "robustness_gap.png", dpi=150)
    plt.close()
    print("Saved: robustness_gap.png")


def plot_token_efficiency(acc_df):
    """Plot token efficiency (accuracy per 100 tokens) by budget."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in acc_df["dataset"].unique():
        sub = acc_df[acc_df["dataset"] == dataset].sort_values("budget", key=lambda x: x.map({b: i for i, b in enumerate(BUDGET_ORDER)}))
        x = np.arange(len(sub))
        ax.bar(x + list(acc_df["dataset"].unique()).index(dataset) * 0.15 - 0.3,
               sub["token_efficiency"], width=0.15, label=dataset,
               color=DATASET_COLORS.get(dataset, None), alpha=0.8)

    ax.set_xticks(range(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=10)
    ax.set_ylabel("Token Efficiency (acc per 100 tokens)", fontsize=12)
    ax.set_xlabel("Reasoning Budget", fontsize=12)
    ax.set_title("Token Efficiency vs Reasoning Budget", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "token_efficiency.png", dpi=150)
    plt.close()
    print("Saved: token_efficiency.png")


def plot_confidence_calibration(df):
    """Plot confidence vs accuracy for calibration analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: confidence by budget and dataset
    ax = axes[0]
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        means = sub.groupby("budget")["confidence"].mean().reindex(BUDGET_ORDER)
        x = np.arange(len(BUDGET_ORDER))
        ax.plot(x, means.values, marker='o', linewidth=2, label=dataset,
                color=DATASET_COLORS.get(dataset, None))
    ax.set_xticks(range(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=9)
    ax.set_ylabel("Mean Confidence", fontsize=12)
    ax.set_title("Model Confidence by Budget", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: calibration plot (confidence bins vs actual accuracy)
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        bin_accs = []
        for i in range(len(bins) - 1):
            mask = (sub["confidence"] >= bins[i]) & (sub["confidence"] < bins[i+1])
            if mask.sum() > 5:
                bin_accs.append(sub.loc[mask, "correct"].mean())
            else:
                bin_accs.append(np.nan)
        ax.plot(bin_centers, bin_accs, marker='o', linewidth=2, label=dataset,
                color=DATASET_COLORS.get(dataset, None))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Actual Accuracy", fontsize=12)
    ax.set_title("Confidence Calibration", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence_calibration.png", dpi=150)
    plt.close()
    print("Saved: confidence_calibration.png")


def plot_length_distribution(df):
    """Plot actual token length distributions by budget condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_box = []
    labels = []
    for budget in BUDGET_ORDER:
        sub = df[df["budget"] == budget]
        data_for_box.append(sub["completion_tokens"].values)
        labels.append(BUDGET_LABELS[budget])

    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
    colors = ['#fee0d2', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Completion Tokens", fontsize=12)
    ax.set_title("Actual Token Usage by Budget Condition", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "length_distribution.png", dpi=150)
    plt.close()
    print("Saved: length_distribution.png")


def plot_difficulty_interaction(df):
    """Plot accuracy vs budget for MATH difficulty levels."""
    math_df = df[df["dataset"] == "MATH"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = sorted(math_df["level"].unique())
    cmap = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(levels)))

    for i, level in enumerate(levels):
        sub = math_df[math_df["level"] == level]
        means = sub.groupby("budget")["correct"].mean().reindex(BUDGET_ORDER)
        x = np.arange(len(BUDGET_ORDER))
        ax.plot(x, means.values, marker='o', linewidth=2, markersize=7,
                label=level, color=cmap[i])

    ax.set_xticks(range(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("MATH Accuracy by Difficulty Level and Budget", fontsize=14)
    ax.legend(fontsize=10, title="Level", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "difficulty_interaction.png", dpi=150)
    plt.close()
    print("Saved: difficulty_interaction.png")


def statistical_tests(df, acc_df):
    """Run statistical tests and return summary."""
    results = {}

    # 1. Spearman correlation: budget index vs accuracy per dataset
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        rho, p = stats.spearmanr(sub["budget_idx"], sub["correct"].astype(float))
        results[f"spearman_{dataset}_budget_acc"] = {"rho": rho, "p": p}

    # 2. McNemar-style: compare no_cot vs medium, medium vs unconstrained per dataset
    for dataset in df["dataset"].unique():
        for pair in [("no_cot", "medium"), ("medium", "unconstrained")]:
            a_correct = df[(df["dataset"] == dataset) & (df["budget"] == pair[0])].set_index("question_id")["correct"]
            b_correct = df[(df["dataset"] == dataset) & (df["budget"] == pair[1])].set_index("question_id")["correct"]
            common = a_correct.index.intersection(b_correct.index)
            if len(common) > 0:
                a = a_correct.loc[common].values.astype(int)
                b = b_correct.loc[common].values.astype(int)
                # Two-sided proportion test
                diff = b.mean() - a.mean()
                n = len(common)
                se = np.sqrt((a.mean() * (1 - a.mean()) + b.mean() * (1 - b.mean())) / n)
                if se > 0:
                    z = diff / se
                    p = 2 * (1 - stats.norm.cdf(abs(z)))
                else:
                    z, p = 0, 1.0
                results[f"prop_test_{dataset}_{pair[0]}_vs_{pair[1]}"] = {
                    "diff": diff, "z": z, "p": p, "n": n
                }

    # 3. Confidence-accuracy correlation
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        rho, p = stats.spearmanr(sub["confidence"], sub["correct"].astype(float))
        results[f"conf_acc_corr_{dataset}"] = {"rho": rho, "p": p}

    # 4. Does robustness gap vary significantly with budget? (Friedman-like)
    # Compare gap across budgets using per-OOD-dataset paired data
    math_acc = acc_df[acc_df["dataset"] == "MATH"].set_index("budget")["accuracy"]
    gap_data = {}
    for dataset in ["GSM8K", "MMLU-Pro", "MuSR"]:
        ood_acc = acc_df[acc_df["dataset"] == dataset].set_index("budget")["accuracy"]
        gap = math_acc - ood_acc
        gap_data[dataset] = gap.reindex(BUDGET_ORDER).values

    results["robustness_gap_summary"] = {
        ds: {b: float(gap_data[ds][i]) for i, b in enumerate(BUDGET_ORDER)}
        for ds in gap_data
    }

    return results


def generate_summary_table(acc_df):
    """Generate a nice markdown summary table."""
    lines = ["| Dataset | No CoT | Short | Medium | Long | Unconstrained |",
             "|---------|--------|-------|--------|------|---------------|"]
    for dataset in ["MATH", "GSM8K", "MMLU-Pro", "MuSR"]:
        row = [dataset]
        for budget in BUDGET_ORDER:
            sub = acc_df[(acc_df["dataset"] == dataset) & (acc_df["budget"] == budget)]
            if len(sub) > 0:
                acc = sub.iloc[0]["accuracy"]
                ci_lo = sub.iloc[0]["ci_lower"]
                ci_hi = sub.iloc[0]["ci_upper"]
                row.append(f"{acc:.1%} [{ci_lo:.1%}-{ci_hi:.1%}]")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    print("Loading results...")
    df = load_results()
    print(f"Total results: {len(df)}")

    print("\nComputing accuracy table...")
    acc_df = compute_accuracy_table(df)
    print(acc_df[["dataset", "budget", "accuracy", "avg_completion_tokens"]].to_string())

    print("\nGenerating plots...")
    plot_accuracy_curves(acc_df)
    plot_actual_token_vs_accuracy(df)
    plot_robustness_gap(acc_df)
    plot_token_efficiency(acc_df)
    plot_confidence_calibration(df)
    plot_length_distribution(df)
    plot_difficulty_interaction(df)

    print("\nRunning statistical tests...")
    stat_results = statistical_tests(df, acc_df)
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2, default=str)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print("\n## Accuracy Table")
    print(generate_summary_table(acc_df))

    print("\n## Spearman Correlations (Budget → Accuracy)")
    for key, val in stat_results.items():
        if key.startswith("spearman"):
            dataset = key.split("_")[1]
            print(f"  {dataset}: ρ={val['rho']:.3f}, p={val['p']:.4f}")

    print("\n## Proportion Tests")
    for key, val in stat_results.items():
        if key.startswith("prop_test"):
            parts = key.replace("prop_test_", "").split("_")
            print(f"  {key}: diff={val['diff']:.3f}, z={val['z']:.3f}, p={val['p']:.4f}")

    print("\n## Confidence-Accuracy Correlations")
    for key, val in stat_results.items():
        if key.startswith("conf_acc"):
            dataset = key.replace("conf_acc_corr_", "")
            print(f"  {dataset}: ρ={val['rho']:.3f}, p={val['p']:.4f}")

    print("\n## Robustness Gap (MATH - OOD) by Budget")
    for ds, gaps in stat_results.get("robustness_gap_summary", {}).items():
        print(f"  vs {ds}:", {b: f"{g:.3f}" for b, g in gaps.items()})

    # Save summary
    summary = {
        "accuracy_table": acc_df.to_dict(orient="records"),
        "markdown_table": generate_summary_table(acc_df),
        "statistical_tests": stat_results,
    }
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAnalysis saved to {RESULTS_DIR}")

    return acc_df, stat_results


if __name__ == "__main__":
    main()
