"""
Model Comparison: Ridge vs XGBoost vs MLP
===========================================
Compares three regression models for predicting tyre degradation rate
(DegSlope) at the stint level.

Methodology:
    - Target: DegSlope (s/lap) from Savitzky-Golay filtered stint data
    - Cross-validation: 5-fold GroupKFold (groups = races)
    - Hyperparameter tuning: GridSearch (Ridge), RandomizedSearch (XGB, MLP)
    - Metrics: MAE, RMSE, R², reported per fold

This produces the key comparison table for the academic report,
answering "why XGBoost over simpler/more complex models?"

Output:
    models/comparison_results.json
    models/best_xgboost_model.json
    models/comparison_summary.txt

Usage:
    python -m src.modeling.model_comparison
"""

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    GroupKFold, GridSearchCV, RandomizedSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint, loguniform

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_stint_data(features_dir: Path, circuit_csv: Path) -> tuple:
    """
    Prepare stint-level training data.
    
    Target: DegSlope (degradation rate in seconds/lap)
    - Already computed via Savitzky-Golay filtering in feature engineering
    - Filters: min 5 laps, dry compounds only, |slope| < 1.0
    """
    logger.info("Preparing stint-level training data...")
    
    stints = pd.read_parquet(features_dir / "stint_features.parquet")
    circuits = pd.read_csv(circuit_csv)
    
    # ── Merge circuit characteristics ──
    circuit_feats = circuits[[
        "season", "round_number",
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution",
        "circuit_length_km", "pit_loss_seconds", "total_laps",
    ]].rename(columns={"season": "Season", "round_number": "RoundNumber"})
    
    data = stints.merge(circuit_feats, on=["Season", "RoundNumber"], how="left")
    
    # ── Merge weather from lap features ──
    lap_features = pd.read_parquet(features_dir / "lap_features.parquet")
    weather = lap_features.groupby(["Season", "RoundNumber"]).agg(
        MeanTrackTemp=("MeanTrackTemp", "first"),
        MeanAirTemp=("MeanAirTemp", "first"),
        MeanHumidity=("MeanHumidity", "first"),
        MeanWindSpeed=("MeanWindSpeed", "first"),
        TrackTempRange=("TrackTempRange", "first"),
    ).reset_index()
    data = data.merge(weather, on=["Season", "RoundNumber"], how="left")
    
    # ── Filter ──
    valid_compounds = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        (data["Compound"].isin(valid_compounds)) &
        (data["StintLength"] >= 5) &
        (data["DegSlope"].abs() < 1.0) &
        (data["DegSlope"].notna())
    )
    data = data[mask].copy()
    
    # ── Feature columns ──
    feature_cols = [
        # Tyre state
        "CompoundHardness", "StintNumber", "StintLength", "TyreLifeStart",
        # Circuit
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution", "circuit_length_km",
        "pit_loss_seconds", "total_laps",
        # Weather
        "MeanTrackTemp", "MeanAirTemp", "MeanHumidity",
        "MeanWindSpeed", "TrackTempRange",
    ]
    
    available = [c for c in feature_cols if c in data.columns]
    
    # Drop rows with NaN in any feature
    for col in available:
        data = data[data[col].notna()]
    
    # Group column for CV (race identity)
    data["RaceGroup"] = data["Season"].astype(str) + "_" + data["RoundNumber"].astype(str)
    
    X = data[available].values
    y = data["DegSlope"].values
    groups = data["RaceGroup"].values
    
    logger.info(f"  Stints: {len(data):,} | Features: {len(available)}")
    logger.info(f"  Target (DegSlope): mean={y.mean():.4f}, std={y.std():.4f}, "
                f"median={np.median(y):.4f}")
    logger.info(f"  Unique races (groups): {len(set(groups))}")
    
    return X, y, groups, available, data


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1: RIDGE REGRESSION (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════

def train_ridge(X, y, groups):
    """Ridge regression with GridSearchCV — simple linear baseline."""
    logger.info("\n" + "─" * 60)
    logger.info("  MODEL 1: Ridge Regression (Baseline)")
    logger.info("─" * 60)
    
    t0 = time.time()
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    
    param_grid = {
        "ridge__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }
    
    cv = GroupKFold(n_splits=5)
    
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        return_train_score=True,
    )
    
    search.fit(X, y, groups=groups)
    
    elapsed = time.time() - t0
    best = search.best_estimator_
    
    logger.info(f"  Best alpha: {search.best_params_['ridge__alpha']}")
    logger.info(f"  Time: {elapsed:.1f}s")
    
    return best, search, elapsed


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2: XGBOOST (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost(X, y, groups):
    """XGBoost with RandomizedSearchCV — primary model."""
    import xgboost as xgb
    
    logger.info("\n" + "─" * 60)
    logger.info("  MODEL 2: XGBoost (Primary)")
    logger.info("─" * 60)
    
    t0 = time.time()
    
    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 8),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(5, 50),
        "reg_alpha": loguniform(0.01, 10.0),
        "reg_lambda": loguniform(0.1, 10.0),
        "gamma": loguniform(0.01, 1.0),
    }
    
    cv = GroupKFold(n_splits=5)
    
    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
        param_distributions,
        n_iter=60,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
    )
    
    search.fit(X, y, groups=groups)
    
    elapsed = time.time() - t0
    
    logger.info(f"  Best params:")
    for k, v in search.best_params_.items():
        logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    logger.info(f"  Time: {elapsed:.1f}s")
    
    return search.best_estimator_, search, elapsed


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 3: MLP NEURAL NETWORK (COMPARISON)
# ═══════════════════════════════════════════════════════════════════════════

def train_mlp(X, y, groups):
    """MLP with RandomizedSearchCV — deep learning comparison."""
    logger.info("\n" + "─" * 60)
    logger.info("  MODEL 3: MLP Neural Network (Comparison)")
    logger.info("─" * 60)
    
    t0 = time.time()
    
    param_distributions = {
        "mlp__hidden_layer_sizes": [
            (64,), (128,), (64, 32), (128, 64),
            (128, 64, 32), (256, 128), (256, 128, 64),
        ],
        "mlp__alpha": loguniform(1e-5, 1e-1),
        "mlp__learning_rate_init": loguniform(1e-4, 1e-2),
        "mlp__batch_size": [32, 64, 128],
    }
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])
    
    cv = GroupKFold(n_splits=5)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=40,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
    )
    
    search.fit(X, y, groups=groups)
    
    elapsed = time.time() - t0
    
    logger.info(f"  Best params:")
    for k, v in search.best_params_.items():
        logger.info(f"    {k}: {v}")
    logger.info(f"  Time: {elapsed:.1f}s")
    
    return search.best_estimator_, search, elapsed


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_per_fold(model, X, y, groups, model_name):
    """Evaluate model per CV fold and return detailed metrics."""
    cv = GroupKFold(n_splits=5)
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and refit
        from sklearn.base import clone
        m = clone(model)
        
        # Handle XGBoost separately (not in pipeline, no groups param)
        try:
            m.fit(X_train, y_train)
        except Exception:
            m.fit(X_train, y_train)
        
        y_pred = m.predict(X_test)
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "test_size": int(len(y_test)),
        })
    
    return fold_metrics


def print_comparison_table(results: dict):
    """Print formatted comparison table."""
    logger.info("\n" + "═" * 72)
    logger.info("  MODEL COMPARISON RESULTS")
    logger.info("═" * 72)
    logger.info(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>8}")
    logger.info(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    
    for name, res in results.items():
        mae = np.mean([f["mae"] for f in res["fold_metrics"]])
        rmse = np.mean([f["rmse"] for f in res["fold_metrics"]])
        r2 = np.mean([f["r2"] for f in res["fold_metrics"]])
        t = res["train_time"]
        
        mae_std = np.std([f["mae"] for f in res["fold_metrics"]])
        r2_std = np.std([f["r2"] for f in res["fold_metrics"]])
        
        logger.info(
            f"  {name:<25} {mae:>7.4f}s {rmse:>7.4f}s {r2:>7.4f} {t:>7.1f}s"
        )
        logger.info(
            f"  {'':25} ±{mae_std:.4f}  {'':8} ±{r2_std:.4f}"
        )
    
    logger.info("═" * 72)
    
    # Determine winner
    model_scores = {}
    for name, res in results.items():
        model_scores[name] = np.mean([f["mae"] for f in res["fold_metrics"]])
    
    winner = min(model_scores, key=model_scores.get)
    logger.info(f"\n  Best model by MAE: {winner} ({model_scores[winner]:.4f}s)")
    
    # Interpretation
    ridge_mae = model_scores.get("Ridge (Baseline)", float("inf"))
    xgb_mae = model_scores.get("XGBoost (Primary)", float("inf"))
    mlp_mae = model_scores.get("MLP (Neural Net)", float("inf"))
    
    if xgb_mae < ridge_mae:
        improvement = (ridge_mae - xgb_mae) / ridge_mae * 100
        logger.info(f"  XGBoost improves {improvement:.1f}% over Ridge baseline")
    
    if mlp_mae < xgb_mae:
        improvement = (xgb_mae - mlp_mae) / xgb_mae * 100
        logger.info(f"  MLP improves {improvement:.1f}% over XGBoost")
    elif xgb_mae <= mlp_mae:
        logger.info(f"  XGBoost matches/beats MLP — confirms tree models' "
                    f"advantage on tabular data")


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (XGBoost)
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_importance(model, feature_names):
    """Extract feature importance from XGBoost model."""
    # Handle case where model might be a pipeline
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    else:
        return {}
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    
    features_dir = Path(config["paths"]["features"])
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 72)
    logger.info("  MODEL COMPARISON: Ridge vs XGBoost vs MLP")
    logger.info("  Target: Tyre Degradation Rate (DegSlope, s/lap)")
    logger.info("  Level: Stint (one prediction per driver-stint)")
    logger.info("  CV: 5-fold GroupKFold (grouped by race)")
    logger.info("=" * 72)
    
    # Prepare data
    X, y, groups, feature_cols, data = prepare_stint_data(features_dir, circuit_csv)
    
    results = {}
    
    # ── Train all three models ──
    ridge_model, ridge_search, ridge_time = train_ridge(X, y, groups)
    ridge_folds = evaluate_per_fold(ridge_model, X, y, groups, "Ridge")
    results["Ridge (Baseline)"] = {
        "fold_metrics": ridge_folds,
        "train_time": ridge_time,
        "best_params": ridge_search.best_params_,
        "cv_best_score": float(-ridge_search.best_score_),
    }
    
    xgb_model, xgb_search, xgb_time = train_xgboost(X, y, groups)
    xgb_folds = evaluate_per_fold(xgb_model, X, y, groups, "XGBoost")
    results["XGBoost (Primary)"] = {
        "fold_metrics": xgb_folds,
        "train_time": xgb_time,
        "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) 
                           else int(v) if isinstance(v, (int, np.integer)) 
                           else v)
                       for k, v in xgb_search.best_params_.items()},
        "cv_best_score": float(-xgb_search.best_score_),
        "feature_importance": get_feature_importance(xgb_model, feature_cols),
    }
    
    mlp_model, mlp_search, mlp_time = train_mlp(X, y, groups)
    mlp_folds = evaluate_per_fold(mlp_model, X, y, groups, "MLP")
    results["MLP (Neural Net)"] = {
        "fold_metrics": mlp_folds,
        "train_time": mlp_time,
        "best_params": {k: str(v) for k, v in mlp_search.best_params_.items()},
        "cv_best_score": float(-mlp_search.best_score_),
    }
    
    # ── Print comparison ──
    print_comparison_table(results)
    
    # ── Save best XGBoost model ──
    xgb_model.save_model(str(model_dir / "best_xgboost_model.json"))
    logger.info(f"\n  ✓ Best XGBoost model saved: models/best_xgboost_model.json")
    
    # ── Save results ──
    output = {
        "experiment": {
            "target": "DegSlope (tyre degradation rate, s/lap)",
            "level": "stint",
            "cv_strategy": "5-fold GroupKFold (grouped by race)",
            "n_samples": int(len(y)),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "results": results,
    }
    
    with open(model_dir / "comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"  ✓ Results saved: models/comparison_results.json")
    
    # ── Human-readable summary ──
    summary_lines = [
        "MODEL COMPARISON SUMMARY",
        "=" * 50,
        f"Target: DegSlope (s/lap) | Samples: {len(y)} stints",
        f"CV: 5-fold GroupKFold | Features: {len(feature_cols)}",
        "",
        f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}",
        f"{'─'*25} {'─'*8} {'─'*8} {'─'*8}",
    ]
    
    for name, res in results.items():
        mae = np.mean([f["mae"] for f in res["fold_metrics"]])
        rmse = np.mean([f["rmse"] for f in res["fold_metrics"]])
        r2 = np.mean([f["r2"] for f in res["fold_metrics"]])
        summary_lines.append(f"{name:<25} {mae:>7.4f}s {rmse:>7.4f}s {r2:>7.4f}")
    
    summary_lines.append("")
    if "feature_importance" in results.get("XGBoost (Primary)", {}):
        summary_lines.append("Top XGBoost Features:")
        for feat, imp in list(results["XGBoost (Primary)"]["feature_importance"].items())[:10]:
            summary_lines.append(f"  {feat:<30} {imp:.4f}")
    
    summary_text = "\n".join(summary_lines)
    with open(model_dir / "comparison_summary.txt", "w") as f:
        f.write(summary_text)
    logger.info(f"  ✓ Summary saved: models/comparison_summary.txt")


if __name__ == "__main__":
    run_comparison()
