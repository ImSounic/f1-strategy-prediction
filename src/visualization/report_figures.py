"""
Report Figures Generator
==========================
Generates all figures for the final report in a consistent style.

Figures:
    1. Rolling validation learning curve (THE headline figure)
    2. Model comparison bar chart (Ridge vs XGBoost vs MLP)
    3. Feature importance (XGBoost gain vs SHAP comparison)
    4. Strategy comparison: Bahrain vs Monaco
    5. Monte Carlo distribution histograms
    6. Degradation curves by compound
    7. Circuit clustering map
    8. Safety car Bayesian priors
    9. DTW stint similarity example

Usage:
    python -m src.visualization.report_figures
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Consistent style ──
COLORS = {
    "primary": "#E10600",     # F1 red
    "secondary": "#1E1E1E",   # Dark
    "accent1": "#0090D0",     # Blue
    "accent2": "#00D2BE",     # Teal (Mercedes-ish)
    "accent3": "#FF8700",     # Orange (McLaren-ish)
    "accent4": "#6B5B95",     # Purple
    "soft": "#FF3333",
    "medium": "#FFC300",
    "hard": "#CCCCCC",
    "bg": "#FFFFFF",
    "grid": "#E8E8E8",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": COLORS["grid"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Rolling Validation Learning Curve
# ═══════════════════════════════════════════════════════════════════════════

def fig_rolling_validation(output_dir):
    """The headline figure showing model improvement with more data."""
    with open("results/validation_rolling_report.json") as f:
        report = json.load(f)

    folds = report["folds"]
    labels = ["2022→2023", "2022-23→2024", "2022-24→2025"]
    n_train = [f["n_training_stints"] for f in folds]
    dry_exact = [f["dry_races"]["exact_rate"] * 100 for f in folds]
    dry_top5 = [f["dry_races"]["top5_rate"] * 100 for f in folds]
    cv_mae = [f["cv_mae"] for f in folds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: accuracy
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax1.bar(x - w/2, dry_exact, w, label="Exact Match", color=COLORS["primary"], zorder=3)
    bars2 = ax1.bar(x + w/2, dry_top5, w, label="Top 5 Match", color=COLORS["accent1"], zorder=3)

    ax1.set_xlabel("Validation Fold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Strategy Prediction Accuracy (Dry Races)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Right: MAE vs training size
    ax2.plot(n_train, cv_mae, "o-", color=COLORS["primary"], linewidth=2, markersize=10, zorder=3)
    for i, (nt, mae) in enumerate(zip(n_train, cv_mae)):
        ax2.annotate(f"{mae:.4f}s", (nt, mae), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    ax2.set_xlabel("Training Stints")
    ax2.set_ylabel("CV MAE (seconds/lap)")
    ax2.set_title("Model Error vs Training Data Size")
    ax2.set_xlim(500, 3500)

    fig.suptitle("Rolling Temporal Validation — No Data Leakage", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_rolling_validation.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_rolling_validation.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_model_comparison(output_dir):
    with open("models/comparison_results.json") as f:
        comp = json.load(f)

    models_data = comp["results"]
    names = list(models_data.keys())

    mae_vals = []
    mae_stds = []
    times = []
    for n in names:
        folds = models_data[n]["fold_metrics"]
        maes = [f["mae"] for f in folds]
        mae_vals.append(float(np.mean(maes)))
        mae_stds.append(float(np.std(maes)))
        times.append(models_data[n]["train_time"])

    display_names = {
        "Ridge (Baseline)": "Ridge\n(Baseline)",
        "XGBoost (Primary)": "XGBoost\n(Primary)",
        "MLP (Neural Net)": "MLP\n(Neural Net)",
    }
    colors = [COLORS["accent4"], COLORS["primary"], COLORS["accent3"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(names))
    bars = ax1.bar(x, mae_vals, yerr=mae_stds, capsize=5,
                   color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax1.set_ylabel("MAE (seconds/lap)")
    ax1.set_title("Cross-Validated MAE (lower = better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([display_names.get(n, n) for n in names])

    for bar, val in zip(bars, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Highlight best
    best_idx = np.argmin(mae_vals)
    bars[best_idx].set_edgecolor(COLORS["primary"])
    bars[best_idx].set_linewidth(3)

    # Right: training time
    ax2.barh(x, times, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax2.set_xlabel("Training Time (seconds)")
    ax2.set_title("Training Time")
    ax2.set_yticks(x)
    ax2.set_yticklabels([display_names.get(n, n) for n in names])

    for i, t in enumerate(times):
        ax2.text(t + 0.3, i, f"{t:.1f}s", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3: Feature Importance (SHAP)
# ═══════════════════════════════════════════════════════════════════════════

def fig_feature_importance(output_dir):
    """SHAP-based feature importance (already have shap_bar.png and shap_summary.png,
    but let's make a cleaner version)."""
    shap_df = pd.read_parquet("results/shap_values.parquet")
    feature_cols = [c for c in shap_df.columns if c not in ["DegSlope_actual", "Compound", "EventName"]]

    mean_abs = shap_df[feature_cols].abs().mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_list = [COLORS["primary"] if v > mean_abs.quantile(0.7) else COLORS["accent1"]
                   for v in mean_abs.values]

    bars = ax.barh(range(len(mean_abs)), mean_abs.values, color=colors_list, zorder=3)
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels(mean_abs.index, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (impact on degradation prediction)")
    ax.set_title("Feature Importance — SHAP Analysis", fontsize=14, fontweight="bold")

    # Add value labels
    for i, v in enumerate(mean_abs.values):
        if v > 0.001:
            ax.text(v + 0.0002, i, f"{v:.4f}", va="center", fontsize=9)

    legend_elements = [
        Patch(facecolor=COLORS["primary"], label="High importance"),
        Patch(facecolor=COLORS["accent1"], label="Lower importance"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Strategy Comparison (Bahrain vs Monaco)
# ═══════════════════════════════════════════════════════════════════════════

def fig_strategy_comparison(output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, circuit_file, title in zip(
        axes,
        ["strategy_bahrain_2024.json", "strategy_monaco_2024.json"],
        ["Bahrain GP 2024 (High Degradation)", "Monaco GP 2024 (Low Degradation)"],
    ):
        path = Path("results") / circuit_file
        if not path.exists():
            continue

        with open(path) as f:
            data = json.load(f)

        rankings = data["rankings"][:10]
        names = [r["strategy_name"].replace("1-stop ", "1s ").replace("2-stop ", "2s ")
                 for r in rankings]
        medians = [r["median_time"] for r in rankings]
        stds = [r["std_time"] for r in rankings]

        best = medians[0]
        deltas = [m - best for m in medians]

        colors_list = [COLORS["primary"] if r["num_stops"] == 1 else COLORS["accent1"]
                      for r in rankings]

        bars = ax.barh(range(len(names)-1, -1, -1), deltas, xerr=stds,
                      capsize=3, color=colors_list, zorder=3, alpha=0.85)
        ax.set_yticks(range(len(names)-1, -1, -1))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Delta to Best (seconds)")
        ax.set_title(title, fontsize=12, fontweight="bold")

        legend_elements = [
            Patch(facecolor=COLORS["primary"], label="1-stop"),
            Patch(facecolor=COLORS["accent1"], label="2-stop"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_strategy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_strategy_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 5: Monte Carlo Distribution
# ═══════════════════════════════════════════════════════════════════════════

def fig_monte_carlo_distribution(output_dir):
    """Show distribution of race times for top strategies at a circuit."""
    # We need to re-run a few sims to get actual distributions
    # Instead, show the P5-P95 range as a forest plot

    with open("results/strategy_bahrain_2024.json") as f:
        data = json.load(f)

    top5 = data["rankings"][:5]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, r in enumerate(top5):
        y = len(top5) - 1 - i
        median = r["median_time"]
        p5 = r["p5_time"]
        p95 = r["p95_time"]
        mean = r["mean_time"]

        # P5-P95 range
        ax.plot([p5, p95], [y, y], color=COLORS["accent1"], linewidth=3, alpha=0.6, zorder=2)
        # Median marker
        ax.plot(median, y, "D", color=COLORS["primary"], markersize=10, zorder=3)
        # Mean marker
        ax.plot(mean, y, "o", color=COLORS["accent3"], markersize=7, zorder=3)

    labels = [r["strategy_name"] for r in top5]
    ax.set_yticks(range(len(labels)-1, -1, -1))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Total Race Time (seconds)")
    ax.set_title("Bahrain GP 2024 — Monte Carlo Strategy Distributions (1000 sims each)",
                fontsize=12, fontweight="bold")

    legend_elements = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["primary"],
                   markersize=10, label="Median"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["accent3"],
                   markersize=8, label="Mean"),
        plt.Line2D([0], [0], color=COLORS["accent1"], linewidth=3, alpha=0.6, label="P5–P95 range"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_monte_carlo_dist.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_monte_carlo_dist.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 6: Degradation Curves by Compound
# ═══════════════════════════════════════════════════════════════════════════

def fig_degradation_curves(output_dir):
    config = load_config()
    features_dir = Path(config["paths"]["features"])
    laps = pd.read_parquet(features_dir / "lap_features.parquet")

    valid = {"C1", "C2", "C3", "C4", "C5"}
    clean = laps[(laps["IsClean"] == True) & (laps["ActualCompound"].isin(valid))].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    compound_colors = {"C1": "#CCCCCC", "C2": "#AAAAAA", "C3": "#FFC300", "C4": "#FF8700", "C5": "#FF3333"}
    compound_names = {"C1": "C1 (Hardest)", "C2": "C2", "C3": "C3 (Medium)", "C4": "C4", "C5": "C5 (Softest)"}

    for compound in ["C1", "C2", "C3", "C4", "C5"]:
        subset = clean[clean["ActualCompound"] == compound]

        # Group by tyre life and compute median fuel-corrected time delta
        grouped = subset.groupby("TyreLife").agg(
            median_delta=("FuelCorrectedLapTime", "median"),
            count=("FuelCorrectedLapTime", "count"),
        )

        # Only show where we have enough data
        grouped = grouped[grouped["count"] >= 20]
        if grouped.empty:
            continue

        # Normalize to first lap
        baseline = grouped.iloc[0]["median_delta"]
        grouped["normalized"] = grouped["median_delta"] - baseline

        ax.plot(grouped.index, grouped["normalized"],
               color=compound_colors[compound], linewidth=2.5,
               label=compound_names[compound], zorder=3)

    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("Lap Time Delta from Fresh (seconds)")
    ax.set_title("Tyre Degradation Curves by Compound (2022-2025, all circuits)",
                fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 40)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_degradation_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_degradation_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 7: Safety Car Bayesian Priors
# ═══════════════════════════════════════════════════════════════════════════

def fig_safety_car_priors(output_dir):
    with open("models/safety_car_priors.json") as f:
        priors = json.load(f)

    circuits = []
    raw_probs = []
    bayes_probs = []

    for key, val in sorted(priors.items(), key=lambda x: x[1].get("bayesian_sc_prob", 0)):
        circuits.append(key.replace("_", " ").title()[:20])
        raw_probs.append(val.get("raw_sc_rate", 0) * 100)
        bayes_probs.append(val.get("bayesian_sc_prob", 0) * 100)

    fig, ax = plt.subplots(figsize=(12, 7))

    y = np.arange(len(circuits))
    w = 0.35

    ax.barh(y + w/2, raw_probs, w, label="Raw Historical Rate", color=COLORS["accent1"], alpha=0.6, zorder=2)
    ax.barh(y - w/2, bayes_probs, w, label="Bayesian Posterior", color=COLORS["primary"], zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(circuits, fontsize=8)
    ax.set_xlabel("Safety Car Probability (%)")
    ax.set_title("Bayesian Safety Car Probabilities per Circuit\n(Beta-Binomial with shrinkage toward global mean)",
                fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.axvline(x=55.4, color="gray", linestyle="--", alpha=0.5, label="Global mean (55.4%)")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_safety_car_priors.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_safety_car_priors.png")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 8: System Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════

def fig_system_architecture(output_dir):
    """Simple pipeline diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, "Data\nIngestion", COLORS["accent1"]),
        (3.0, 1.5, "Feature\nEngineering", COLORS["accent2"]),
        (5.5, 1.5, "Model\nTraining", COLORS["primary"]),
        (8.0, 1.5, "Monte Carlo\nSimulator", COLORS["accent3"]),
        (10.5, 1.5, "Strategy\nRecommendation", COLORS["accent4"]),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x, y), 2.0, 1.5, facecolor=color, edgecolor="white",
                            linewidth=2, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(x + 1.0, y + 0.75, text, ha="center", va="center",
               fontsize=10, fontweight="bold", color="white", zorder=4)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 2.0
        x2 = boxes[i+1][0]
        y = 2.25
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=2))

    # Sub-labels
    sub_labels = [
        (1.5, 1.1, "FastF1 · OpenF1\nJolpica · Pirelli", 8),
        (4.0, 1.1, "Savitzky-Golay\nFuel Correction", 8),
        (6.5, 1.1, "XGBoost · Bayesian\nClustering · DTW", 8),
        (9.0, 1.1, "1000 sims/strategy\nStochastic SC", 8),
        (11.5, 1.1, "71% exact match\n86% top-5 (2025)", 8),
    ]
    for x, y, text, size in sub_labels:
        ax.text(x, y, text, ha="center", va="top", fontsize=size, color=COLORS["secondary"], style="italic")

    ax.set_title("F1 Race Strategy Optimization — System Pipeline", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_system_architecture.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ fig_system_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  GENERATING REPORT FIGURES")
    logger.info("=" * 60)

    fig_rolling_validation(output_dir)
    fig_model_comparison(output_dir)
    fig_feature_importance(output_dir)
    fig_strategy_comparison(output_dir)
    fig_monte_carlo_distribution(output_dir)
    fig_degradation_curves(output_dir)
    fig_safety_car_priors(output_dir)
    fig_system_architecture(output_dir)

    logger.info(f"\n  ✓ All figures saved to {output_dir}/")
    logger.info(f"  Total: {len(list(output_dir.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
