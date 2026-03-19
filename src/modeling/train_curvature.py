"""
Curvature Model Training
==========================
Trains a separate XGBoost model to predict DegCurvature (quadratic coefficient
of tyre degradation) using the same features as the DegSlope model.

The curvature captures how degradation accelerates through a stint:
  - curvature > 0: degradation speeds up (cliff approaching)
  - curvature ~ 0: linear degradation
  - curvature < 0: degradation slows (tyre warming up)

Used alongside the DegSlope model in simulation for non-linear tyre curves:
  tyre_deg = slope * age + curvature * age²

Output:
    models/tyre_curvature_model.json
    models/curvature_results.json

Usage:
    python -m src.modeling.train_curvature
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform, randint, loguniform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_curvature_model(config_path: str = "configs/config.yaml"):
    config = yaml.safe_load(open(config_path))
    features_dir = Path(config["paths"]["features"])
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"

    # Load data (same pipeline as DegSlope model)
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

    # Filter
    valid = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        data["Compound"].isin(valid) &
        (data["StintLength"] >= 5) &
        data["DegCurvature"].notna() &
        (data["DegCurvature"].abs() < 0.5)
    )
    data = data[mask].copy()

    # Use SAME feature columns as the slope model for consistency
    feature_cols = [
        "CompoundHardness", "StintNumber", "StintLength", "TyreLifeStart",
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution", "circuit_length_km",
        "pit_loss_seconds",
        "MeanTrackTemp", "MeanAirTemp", "MeanHumidity",
        "MeanWindSpeed", "TrackTempRange",
    ]
    available = [c for c in feature_cols if c in data.columns]
    data = data.dropna(subset=available + ["DegCurvature"])

    X = data[available].values
    y = data["DegCurvature"].values
    groups = data["Season"].astype(str) + "_" + data["RoundNumber"].astype(str)

    logger.info("=" * 60)
    logger.info("  CURVATURE MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Target: DegCurvature (quadratic coefficient)")
    logger.info(f"  Mean: {y.mean():.6f}, Std: {y.std():.6f}, Median: {np.median(y):.6f}")

    # Train XGBoost
    t0 = time.time()
    cv = GroupKFold(n_splits=min(5, len(set(groups))))

    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
        {
            "n_estimators": randint(100, 400),
            "max_depth": randint(3, 7),
            "learning_rate": loguniform(0.01, 0.2),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": randint(10, 60),
            "reg_alpha": loguniform(0.1, 10.0),
            "reg_lambda": loguniform(0.1, 10.0),
        },
        n_iter=40, cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1,
    )
    search.fit(X, y, groups=groups.values)

    model = search.best_estimator_
    cv_mae = -search.best_score_
    elapsed = time.time() - t0

    # Per-fold evaluation
    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups.values)):
        from sklearn.base import clone
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])
        fold_metrics.append({
            "fold": fold_idx + 1,
            "mae": float(mean_absolute_error(y[test_idx], y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[test_idx], y_pred))),
            "r2": float(r2_score(y[test_idx], y_pred)),
        })

    avg_mae = np.mean([f["mae"] for f in fold_metrics])
    avg_r2 = np.mean([f["r2"] for f in fold_metrics])

    logger.info(f"\n  CV MAE: {avg_mae:.6f}")
    logger.info(f"  CV R²:  {avg_r2:.4f}")
    logger.info(f"  Time:   {elapsed:.1f}s")

    # Save model
    model_dir = Path("models")
    model.save_model(str(model_dir / "tyre_curvature_model.json"))
    logger.info(f"\n  ✓ Model saved: models/tyre_curvature_model.json")

    # Save results
    output = {
        "target": "DegCurvature (quadratic coefficient of tyre degradation)",
        "n_samples": int(len(X)),
        "feature_columns": available,
        "cv_mae": float(cv_mae),
        "fold_metrics": fold_metrics,
        "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in search.best_params_.items()},
    }
    with open(model_dir / "curvature_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  ✓ Results saved: models/curvature_results.json")


if __name__ == "__main__":
    train_curvature_model()
