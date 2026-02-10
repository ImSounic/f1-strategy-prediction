"""
SHAP Analysis for Model Interpretability
==========================================
Generates SHAP (SHapley Additive exPlanations) analysis for the
tyre degradation XGBoost model.

Produces:
    - Global feature importance (SHAP bar plot)
    - SHAP summary/beeswarm plot (shows direction of effects)
    - Dependence plots for top features
    - Interaction effects

Output:
    results/shap_summary.png
    results/shap_bar.png
    results/shap_dependence_*.png
    results/shap_values.parquet

Usage:
    python -m src.analysis.shap_analysis
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(features_dir: Path, circuit_csv: Path, feature_cols: list):
    """Load stint data for SHAP analysis (same prep as model_comparison)."""
    stints = pd.read_parquet(features_dir / "stint_features.parquet")
    circuits = pd.read_csv(circuit_csv)
    
    circuit_feats = circuits[[
        "season", "round_number",
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution",
        "circuit_length_km", "pit_loss_seconds", "total_laps",
    ]].rename(columns={"season": "Season", "round_number": "RoundNumber"})
    
    data = stints.merge(circuit_feats, on=["Season", "RoundNumber"], how="left")
    
    # Weather
    lap_features = pd.read_parquet(features_dir / "lap_features.parquet")
    weather = lap_features.groupby(["Season", "RoundNumber"]).agg(
        MeanTrackTemp=("MeanTrackTemp", "first"),
        MeanAirTemp=("MeanAirTemp", "first"),
        MeanHumidity=("MeanHumidity", "first"),
        MeanWindSpeed=("MeanWindSpeed", "first"),
        TrackTempRange=("TrackTempRange", "first"),
    ).reset_index()
    data = data.merge(weather, on=["Season", "RoundNumber"], how="left")
    
    valid = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        data["Compound"].isin(valid) &
        (data["StintLength"] >= 5) &
        (data["DegSlope"].abs() < 1.0) &
        data["DegSlope"].notna()
    )
    data = data[mask].copy()
    
    available = [c for c in feature_cols if c in data.columns]
    data = data.dropna(subset=available + ["DegSlope"])
    
    return data, available


def run_shap_analysis(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    
    features_dir = Path(config["paths"]["features"])
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  SHAP ANALYSIS")
    logger.info("=" * 60)
    
    # Load model and feature columns
    model = xgb.XGBRegressor()
    model.load_model("models/best_xgboost_model.json")
    
    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]
    
    # Load data
    data, available = prepare_data(features_dir, circuit_csv, feature_cols)
    X = data[available]
    
    logger.info(f"  Samples: {len(X):,} | Features: {len(available)}")
    
    # ── Compute SHAP values ──
    logger.info("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    logger.info(f"  SHAP values shape: {shap_values.shape}")
    
    # ── Plot 1: SHAP Bar Plot (global importance) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ shap_bar.png")
    
    # ── Plot 2: SHAP Beeswarm/Summary Plot ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("SHAP Summary: Feature Effects on Degradation Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ shap_summary.png")
    
    # ── Plot 3: Dependence plots for top 4 features ──
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:4]
    
    for idx in top_indices:
        feat_name = available[idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values, X, show=False, ax=ax)
        ax.set_title(f"SHAP Dependence: {feat_name}", fontsize=13)
        plt.tight_layout()
        safe_name = feat_name.replace("/", "_")
        plt.savefig(output_dir / f"shap_dep_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ shap_dep_{safe_name}.png")
    
    # ── Save SHAP values for further analysis ──
    shap_df = pd.DataFrame(shap_values, columns=available)
    shap_df["DegSlope_actual"] = data["DegSlope"].values
    shap_df["Compound"] = data["Compound"].values
    shap_df["EventName"] = data["EventName"].values
    shap_df.to_parquet(output_dir / "shap_values.parquet", index=False)
    logger.info(f"  ✓ shap_values.parquet ({len(shap_df):,} rows)")
    
    # ── Print top SHAP features ──
    logger.info("\n  SHAP Feature Ranking (mean |SHAP|):")
    ranking = sorted(zip(available, mean_abs_shap), key=lambda x: x[1], reverse=True)
    for feat, val in ranking:
        bar = "█" * int(val * 200)
        logger.info(f"    {feat:<30} {val:.5f} {bar}")
    
    logger.info("\n  ✓ SHAP analysis complete")


if __name__ == "__main__":
    run_shap_analysis()
