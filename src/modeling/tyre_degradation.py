"""
Tyre Degradation Model v2 (XGBoost Regression)
================================================
FIXED: Target is now within-stint degradation — predicting how much
slower a lap is relative to the driver's own stint opening pace.

Target: LapTimeDeltaToStint = FuelCorrectedLapTime - StintBaseline
    where StintBaseline = median of first 3 clean laps in the stint

This isolates tyre degradation from driver/car performance differences.

Output:
    models/tyre_degradation_model.json
    models/tyre_deg_evaluation.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_training_data(
    features_dir: Path,
    circuit_csv: Path,
) -> tuple[pd.DataFrame, list]:
    """
    Prepare lap-level training data with within-stint target.
    
    Key change from v1: instead of predicting delta to race median (which
    confounds driver/car pace), predict delta to the driver's own stint
    opening pace. This isolates the tyre degradation signal.
    """
    logger.info("Preparing training data (v2: within-stint target)...")
    
    laps = pd.read_parquet(features_dir / "lap_features.parquet")
    circuits = pd.read_csv(circuit_csv)
    
    # ── Compute stint baseline: median of first 3 laps per stint ──
    laps = laps.sort_values(["Season", "RoundNumber", "Driver", "LapNumber"])
    
    stint_groups = laps.groupby(["Season", "RoundNumber", "Driver", "StintNumber"])
    
    def stint_baseline(group):
        """Median of first 3 clean laps as the stint's reference pace."""
        first_laps = group.head(3)["FuelCorrectedLapTime"]
        return first_laps.median()
    
    baselines = stint_groups.apply(stint_baseline).rename("StintBaseline")
    baselines = baselines.reset_index()
    
    laps = laps.merge(
        baselines, on=["Season", "RoundNumber", "Driver", "StintNumber"], how="left"
    )
    
    # Target: delta to own stint baseline (positive = degraded)
    laps["LapTimeDeltaToStint"] = laps["FuelCorrectedLapTime"] - laps["StintBaseline"]
    
    # ── Tyre age within stint (more robust than TyreLife which can have issues) ──
    laps["TyreAgeInStint"] = stint_groups.cumcount()
    
    # ── Merge circuit features ──
    circuit_features = circuits[[
        "season", "round_number",
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution",
        "circuit_length_km", "pit_loss_seconds",
    ]].rename(columns={"season": "Season", "round_number": "RoundNumber"})
    
    laps = laps.merge(circuit_features, on=["Season", "RoundNumber"], how="left")
    
    # ── Filter: only clean laps with valid compounds ──
    valid_compounds = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        (laps["IsClean"] == True) &
        (laps["ActualCompound"].isin(valid_compounds)) &
        (laps["LapTimeDeltaToStint"].notna()) &
        (laps["TyreAgeInStint"] >= 1)  # skip the baseline laps themselves
    )
    
    # Remove extreme outliers (>10s from baseline — SC aftermath, etc.)
    mask &= laps["LapTimeDeltaToStint"].abs() < 10.0
    
    data = laps[mask].copy()
    
    # ── Feature columns ──
    feature_cols = [
        # Tyre state (most important)
        "TyreAgeInStint", "CompoundHardness", "StintNumber",
        # Circuit characteristics
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution", "circuit_length_km",
        # Weather
        "MeanTrackTemp", "MeanAirTemp", "MeanHumidity",
        "MeanWindSpeed", "TrackTempRange",
        # Race context
        "LapNumber", "Position",
    ]
    
    available = [c for c in feature_cols if c in data.columns]
    for col in available:
        mask_notna = data[col].notna()
        data = data[mask_notna]
    
    logger.info(f"  Training samples: {len(data):,} laps with {len(available)} features")
    logger.info(f"  Target stats: mean={data['LapTimeDeltaToStint'].mean():.4f}s, "
                f"std={data['LapTimeDeltaToStint'].std():.4f}s, "
                f"median={data['LapTimeDeltaToStint'].median():.4f}s")
    
    return data, available


def train_model(data: pd.DataFrame, feature_cols: list, target_col: str = "LapTimeDeltaToStint"):
    """Train XGBoost with group-based split on races + early stopping."""
    import xgboost as xgb
    
    logger.info("Training XGBoost degradation model (v2)...")
    
    X = data[feature_cols].values
    y = data[target_col].values
    
    groups = data["Season"].astype(str) + "_" + data["RoundNumber"].astype(str)
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_features": len(feature_cols),
        "best_iteration": int(model.best_iteration) if hasattr(model, 'best_iteration') else 500,
    }
    
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    logger.info(f"  MAE:  {metrics['mae']:.4f}s")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}s")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  Best iteration: {metrics['best_iteration']}")
    logger.info(f"  Top features:")
    for feat, imp in list(importance.items())[:7]:
        logger.info(f"    {feat}: {imp:.4f}")
    
    return model, metrics, importance


def run_tyre_degradation_model(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    
    features_dir = Path(config["paths"]["features"])
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  TYRE DEGRADATION MODEL v2 (XGBoost)")
    logger.info("=" * 60)
    
    data, feature_cols = prepare_training_data(features_dir, circuit_csv)
    model, metrics, importance = train_model(data, feature_cols)
    
    model.save_model(str(model_dir / "tyre_degradation_model.json"))
    logger.info(f"  ✓ Model saved: models/tyre_degradation_model.json")
    
    evaluation = {
        "model_version": "v2_within_stint",
        "target": "LapTimeDeltaToStint (delta to stint opening pace)",
        "metrics": metrics,
        "feature_importance": importance,
        "feature_columns": feature_cols,
    }
    with open(model_dir / "tyre_deg_evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    logger.info(f"  ✓ Evaluation saved: models/tyre_deg_evaluation.json")


if __name__ == "__main__":
    run_tyre_degradation_model()
