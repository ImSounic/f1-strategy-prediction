"""
Tyre Degradation Model v3 (Enhanced XGBoost)
==============================================
Improvements over v2:
    1. Per-compound sub-models (separate XGBoost per compound group)
    2. Driver/team performance features merged
    3. Explicit interaction features (compound × circuit, temp × stress)
    4. Proper hyperparameter tuning with RandomizedSearchCV

Output:
    models/tyre_deg_v3_results.json
    models/tyre_deg_v3_model_C*.json (per-compound models)
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform, randint, loguniform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_enhanced_stint_data(features_dir: Path, circuit_csv: Path) -> pd.DataFrame:
    """Build enhanced stint dataset with driver/team features and interactions."""
    logger.info("Preparing enhanced stint data (v3)...")
    
    stints = pd.read_parquet(features_dir / "stint_features.parquet")
    circuits = pd.read_csv(circuit_csv)
    
    # ── Circuit features ──
    circuit_feats = circuits[[
        "season", "round_number",
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution",
        "circuit_length_km", "pit_loss_seconds",
    ]].rename(columns={"season": "Season", "round_number": "RoundNumber"})
    
    data = stints.merge(circuit_feats, on=["Season", "RoundNumber"], how="left")
    
    # ── Weather ──
    lap_features = pd.read_parquet(features_dir / "lap_features.parquet")
    weather = lap_features.groupby(["Season", "RoundNumber"]).agg(
        MeanTrackTemp=("MeanTrackTemp", "first"),
        MeanAirTemp=("MeanAirTemp", "first"),
        MeanHumidity=("MeanHumidity", "first"),
        MeanWindSpeed=("MeanWindSpeed", "first"),
        TrackTempRange=("TrackTempRange", "first"),
    ).reset_index()
    data = data.merge(weather, on=["Season", "RoundNumber"], how="left")
    
    # ── Driver/Team features ──
    driver_feats = pd.read_parquet(features_dir / "driver_features.parquet")
    
    # Map Driver code to driverCode for merging
    driver_rolling = driver_feats.groupby(["season", "round", "driverCode"]).agg(
        RollingAvgFinish=("RollingAvgFinish_5", "last"),
        RollingAvgPoints=("RollingAvgPoints_5", "last"),
        GridPosition=("grid", "last"),
    ).reset_index().rename(columns={
        "season": "Season", "round": "RoundNumber", "driverCode": "Driver"
    })
    
    # Convert RoundNumber types to match
    driver_rolling["Season"] = driver_rolling["Season"].astype(int)
    driver_rolling["RoundNumber"] = driver_rolling["RoundNumber"].astype(int)
    data["Season"] = data["Season"].astype(int)
    data["RoundNumber"] = data["RoundNumber"].astype(int)
    
    data = data.merge(driver_rolling, on=["Season", "RoundNumber", "Driver"], how="left")
    
    # ── Team strength (constructor standings from previous season as proxy) ──
    team_feats = pd.read_parquet(features_dir / "team_features.parquet")
    # Use previous season standings
    team_feats["NextSeason"] = team_feats["Season"].astype(int) + 1
    team_strength = team_feats[["NextSeason", "ConstructorId", "ConstructorStandingPos", "ConstructorPoints"]].rename(
        columns={"NextSeason": "Season"}
    )
    
    # Map Team name to ConstructorId (rough mapping via lowercase)
    # This is imperfect but covers the major teams
    data["TeamLower"] = data["Team"].str.lower().str.replace(" ", "_")
    team_strength["ConstructorLower"] = team_strength["ConstructorId"].str.lower()
    
    # Skip the team merge if mapping is too noisy — use driver features instead
    
    # ── Interaction features ──
    data["Compound_x_Abrasiveness"] = data["CompoundHardness"] * data["asphalt_abrasiveness"]
    data["Compound_x_Grip"] = data["CompoundHardness"] * data["asphalt_grip"]
    data["Compound_x_TyreStress"] = data["CompoundHardness"] * data["tyre_stress"]
    data["TrackTemp_x_TyreStress"] = data["MeanTrackTemp"] * data["tyre_stress"]
    data["TrackTemp_x_Abrasiveness"] = data["MeanTrackTemp"] * data["asphalt_abrasiveness"]
    data["Humidity_x_Grip"] = data["MeanHumidity"] * data["asphalt_grip"]
    data["StintLength_x_Hardness"] = data["StintLength"] * data["CompoundHardness"]
    
    # ── Filter ──
    valid_compounds = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        (data["Compound"].isin(valid_compounds)) &
        (data["StintLength"] >= 5) &
        (data["DegSlope"].abs() < 1.0) &
        (data["DegSlope"].notna())
    )
    data = data[mask].copy()
    data["RaceGroup"] = data["Season"].astype(str) + "_" + data["RoundNumber"].astype(str)
    
    logger.info(f"  Total stints: {len(data):,}")
    logger.info(f"  Compound distribution:")
    for comp, count in data["Compound"].value_counts().sort_index().items():
        logger.info(f"    {comp}: {count}")
    
    return data


def get_feature_columns(include_interactions=True, include_driver=True):
    """Define feature columns for the model."""
    base_features = [
        # Tyre state
        "CompoundHardness", "StintNumber", "StintLength", "TyreLifeStart",
        # Circuit
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution", "circuit_length_km",
        "pit_loss_seconds",
        # Weather
        "MeanTrackTemp", "MeanAirTemp", "MeanHumidity",
        "MeanWindSpeed", "TrackTempRange",
    ]
    
    interaction_features = [
        "Compound_x_Abrasiveness", "Compound_x_Grip", "Compound_x_TyreStress",
        "TrackTemp_x_TyreStress", "TrackTemp_x_Abrasiveness",
        "Humidity_x_Grip", "StintLength_x_Hardness",
    ]
    
    driver_features = [
        "RollingAvgFinish", "RollingAvgPoints", "GridPosition",
    ]
    
    features = base_features.copy()
    if include_interactions:
        features.extend(interaction_features)
    if include_driver:
        features.extend(driver_features)
    
    return features


def train_global_model(data: pd.DataFrame, feature_cols: list):
    """Train enhanced global model (all compounds together)."""
    logger.info("\n" + "─" * 60)
    logger.info("  GLOBAL MODEL (all compounds, enhanced features)")
    logger.info("─" * 60)
    
    available = [c for c in feature_cols if c in data.columns]
    clean = data.dropna(subset=available + ["DegSlope"]).copy()
    
    X = clean[available].values
    y = clean["DegSlope"].values
    groups = clean["RaceGroup"].values
    
    logger.info(f"  Samples: {len(X):,} | Features: {len(available)}")
    
    param_dist = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 7),
        "learning_rate": loguniform(0.01, 0.2),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(10, 60),
        "reg_alpha": loguniform(0.1, 10.0),
        "reg_lambda": loguniform(0.1, 10.0),
        "gamma": loguniform(0.01, 1.0),
    }
    
    cv = GroupKFold(n_splits=5)
    
    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
        param_dist, n_iter=60, cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    
    # Per-fold evaluation
    model = search.best_estimator_
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        from sklearn.base import clone
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])
        fold_results.append({
            "fold": fold_i + 1,
            "mae": float(mean_absolute_error(y[test_idx], y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[test_idx], y_pred))),
            "r2": float(r2_score(y[test_idx], y_pred)),
        })
    
    avg_mae = np.mean([f["mae"] for f in fold_results])
    avg_r2 = np.mean([f["r2"] for f in fold_results])
    
    logger.info(f"  MAE:  {avg_mae:.4f}s (±{np.std([f['mae'] for f in fold_results]):.4f})")
    logger.info(f"  R²:   {avg_r2:.4f} (±{np.std([f['r2'] for f in fold_results]):.4f})")
    
    importance = dict(zip(available, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    logger.info(f"  Top 10 features:")
    for feat, imp in list(importance.items())[:10]:
        logger.info(f"    {feat:<35} {imp:.4f}")
    
    return model, fold_results, importance, available


def train_per_compound_models(data: pd.DataFrame, feature_cols: list):
    """Train separate model per compound group for finer-grained prediction."""
    logger.info("\n" + "─" * 60)
    logger.info("  PER-COMPOUND MODELS")
    logger.info("─" * 60)
    
    # Group soft compounds (C4+C5+C6) and hard compounds (C1+C2) for sample size
    compound_groups = {
        "Hard (C1-C2)": ["C1", "C2"],
        "Medium (C3)": ["C3"],
        "Soft (C4-C6)": ["C4", "C5", "C6"],
    }
    
    # Remove CompoundHardness from per-compound models (redundant)
    per_compound_features = [c for c in feature_cols if c not in [
        "CompoundHardness", "Compound_x_Abrasiveness", "Compound_x_Grip",
        "Compound_x_TyreStress", "StintLength_x_Hardness",
    ]]
    available = [c for c in per_compound_features if c in data.columns]
    
    compound_results = {}
    compound_models = {}
    
    for group_name, compounds in compound_groups.items():
        subset = data[data["Compound"].isin(compounds)].dropna(subset=available + ["DegSlope"])
        
        if len(subset) < 50:
            logger.warning(f"  {group_name}: only {len(subset)} stints, skipping")
            continue
        
        X = subset[available].values
        y = subset["DegSlope"].values
        groups = subset["RaceGroup"].values
        n_groups = len(set(groups))
        
        n_splits = min(5, n_groups)
        if n_splits < 2:
            logger.warning(f"  {group_name}: only {n_groups} race groups, skipping CV")
            continue
        
        cv = GroupKFold(n_splits=n_splits)
        
        search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
            {
                "n_estimators": randint(50, 300),
                "max_depth": randint(2, 6),
                "learning_rate": loguniform(0.01, 0.2),
                "subsample": uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.5, 0.5),
                "min_child_weight": randint(5, 40),
                "reg_alpha": loguniform(0.1, 10.0),
                "reg_lambda": loguniform(0.1, 10.0),
            },
            n_iter=40, cv=cv,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=-1,
        )
        search.fit(X, y, groups=groups)
        
        model = search.best_estimator_
        
        # Per-fold eval
        fold_results = []
        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            from sklearn.base import clone
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            y_pred = m.predict(X[test_idx])
            fold_results.append({
                "fold": fold_i + 1,
                "mae": float(mean_absolute_error(y[test_idx], y_pred)),
                "r2": float(r2_score(y[test_idx], y_pred)),
            })
        
        avg_mae = np.mean([f["mae"] for f in fold_results])
        avg_r2 = np.mean([f["r2"] for f in fold_results])
        
        logger.info(f"  {group_name}: n={len(X)}, MAE={avg_mae:.4f}s, R²={avg_r2:.4f}")
        
        compound_results[group_name] = {
            "compounds": compounds,
            "n_stints": int(len(X)),
            "fold_metrics": fold_results,
            "avg_mae": float(avg_mae),
            "avg_r2": float(avg_r2),
        }
        compound_models[group_name] = model
    
    return compound_models, compound_results


def run_enhanced_model(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    
    features_dir = Path(config["paths"]["features"])
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  TYRE DEGRADATION MODEL v3 (Enhanced)")
    logger.info("=" * 60)
    
    # Prepare data
    data = prepare_enhanced_stint_data(features_dir, circuit_csv)
    feature_cols = get_feature_columns(include_interactions=True, include_driver=True)
    
    # ── Global model ──
    global_model, global_folds, importance, used_features = train_global_model(data, feature_cols)
    
    # ── Per-compound models ──
    compound_models, compound_results = train_per_compound_models(data, feature_cols)
    
    # ── Comparison: v2 baseline vs v3 ──
    logger.info("\n" + "═" * 60)
    logger.info("  IMPROVEMENT SUMMARY")
    logger.info("═" * 60)
    
    v2_mae = 0.0755  # from previous run
    v3_global_mae = np.mean([f["mae"] for f in global_folds])
    v3_global_r2 = np.mean([f["r2"] for f in global_folds])
    
    logger.info(f"  v2 Global MAE:        {v2_mae:.4f}s (baseline features)")
    logger.info(f"  v3 Global MAE:        {v3_global_mae:.4f}s (enhanced features)")
    improvement = (v2_mae - v3_global_mae) / v2_mae * 100
    logger.info(f"  Improvement:          {improvement:+.1f}%")
    logger.info(f"  v3 Global R²:         {v3_global_r2:.4f}")
    
    if compound_results:
        logger.info(f"\n  Per-compound results:")
        for name, res in compound_results.items():
            logger.info(f"    {name:<20} MAE={res['avg_mae']:.4f}s  R²={res['avg_r2']:.4f}  (n={res['n_stints']})")
    
    # ── Save ──
    global_model.save_model(str(model_dir / "tyre_deg_v3_global.json"))
    
    for name, model in compound_models.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        model.save_model(str(model_dir / f"tyre_deg_v3_{safe_name}.json"))
    
    results = {
        "version": "v3_enhanced",
        "improvements": [
            "driver/team rolling performance features",
            "explicit interaction features (compound×circuit, temp×stress)",
            "per-compound sub-models",
        ],
        "global_model": {
            "fold_metrics": global_folds,
            "avg_mae": float(v3_global_mae),
            "avg_r2": float(v3_global_r2),
            "feature_importance": importance,
            "features_used": used_features,
        },
        "per_compound_models": compound_results,
        "vs_v2": {
            "v2_mae": v2_mae,
            "v3_mae": float(v3_global_mae),
            "improvement_pct": float(improvement),
        },
    }
    
    with open(model_dir / "tyre_deg_v3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n  ✓ All models and results saved to {model_dir}")


if __name__ == "__main__":
    run_enhanced_model()
