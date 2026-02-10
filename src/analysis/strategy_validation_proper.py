"""
Proper Strategy Validation (No Data Leakage)
===============================================
Temporal holdout: train on past seasons, validate on future season.

    - Model trained on 2022-2023 ONLY
    - Validated against 2024 race winner strategies
    - Model has NEVER seen any 2024 data

This is how a real F1 strategy system would operate: before the
2024 season, you only have historical data from prior seasons.

Output:
    results/validation_proper_report.json

Usage:
    python -m src.analysis.strategy_validation_proper
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
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import uniform, randint, loguniform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.simulation.strategy_simulator import (
    load_circuit_config, generate_strategies, run_monte_carlo,
    COMPOUND_HARDNESS,
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN TEMPORAL MODEL (2022-2023 only)
# ═══════════════════════════════════════════════════════════════════════════

def train_temporal_model(
    features_dir: Path,
    circuit_csv: Path,
    train_seasons: list,
) -> tuple:
    """Train XGBoost on specific seasons only."""
    logger.info(f"  Training model on seasons: {train_seasons}")
    
    stints = pd.read_parquet(features_dir / "stint_features.parquet")
    circuits = pd.read_csv(circuit_csv)
    
    # Circuit features
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
    
    # Filter to training seasons ONLY
    valid = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        data["Compound"].isin(valid) &
        (data["StintLength"] >= 5) &
        (data["DegSlope"].abs() < 1.0) &
        data["DegSlope"].notna() &
        data["Season"].isin(train_seasons)
    )
    data = data[mask].copy()
    
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
    data = data.dropna(subset=available + ["DegSlope"])
    
    X = data[available].values
    y = data["DegSlope"].values
    groups = data["Season"].astype(str) + "_" + data["RoundNumber"].astype(str)
    
    logger.info(f"  Training samples: {len(X):,} stints from {train_seasons}")
    
    # Tune with RandomizedSearchCV
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
    
    logger.info(f"  CV MAE: {cv_mae:.4f}s")
    
    return model, available


# ═══════════════════════════════════════════════════════════════════════════
#  RECONSTRUCT ACTUAL WINNER STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_actual_strategies(
    laps_dir: Path,
    pitstops_path: Path,
    circuits_csv: Path,
    season: int,
) -> dict:
    """Reconstruct what the race winner actually did."""
    results_path = pitstops_path.parent / "results.parquet"
    results = pd.read_parquet(results_path)
    pitstops = pd.read_parquet(pitstops_path)
    circuits = pd.read_csv(circuits_csv)
    
    actual = {}
    
    winners = results[
        (results["season"] == season) & (results["position"] == 1)
    ]
    
    for _, winner in winners.iterrows():
        rnd = int(winner["round"])
        driver_id = winner["driverId"]
        driver_code = winner.get("driverCode", "UNK")
        
        # Pit stops
        pits = pitstops[
            (pitstops["season"] == season) &
            (pitstops["round"] == rnd) &
            (pitstops["driverId"] == driver_id)
        ].sort_values("lap")
        
        n_stops = len(pits)
        pit_laps = pits["lap"].tolist()
        
        # Circuit info
        circuit_row = circuits[
            (circuits["season"] == season) & (circuits["round_number"] == rnd)
        ]
        if circuit_row.empty:
            continue
        
        circuit_name = circuit_row.iloc[0]["circuit_name"]
        circuit_key = circuit_row.iloc[0]["circuit_key"]
        
        # Compounds from FastF1 laps
        compounds = []
        laps_files = list(laps_dir.glob(f"{season}_{rnd:02d}_*_R.parquet"))
        if not laps_files:
            laps_files = list(laps_dir.glob(f"{season}_{rnd}_*_R.parquet"))
        
        if laps_files:
            laps_df = pd.read_parquet(laps_files[0])
            wlaps = laps_df[laps_df["Driver"] == driver_code]
            if not wlaps.empty and "Stint" in wlaps.columns:
                compounds = wlaps.groupby("Stint")["Compound"].first().tolist()
        
        is_wet = any(c in ["INTERMEDIATE", "WET"] for c in compounds)
        
        actual[(season, rnd)] = {
            "circuit_name": circuit_name,
            "circuit_key": circuit_key,
            "winner": driver_code,
            "n_stops": n_stops,
            "pit_laps": pit_laps,
            "compounds": compounds,
            "compound_sequence": " → ".join(compounds) if compounds else "Unknown",
            "is_wet": is_wet,
        }
    
    return actual


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_proper_validation(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    
    raw_paths = config["paths"]["raw"]
    fuel_config = config["modeling"]["fuel_model"]
    features_dir = Path(config["paths"]["features"])
    
    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"
    laps_dir = Path(raw_paths["fastf1"]) / "laps"
    pitstops_path = Path(raw_paths["jolpica"]) / "pitstops.parquet"
    
    logger.info("=" * 70)
    logger.info("  PROPER STRATEGY VALIDATION (Temporal Holdout)")
    logger.info("  Train: 2022-2023 → Validate: 2024")
    logger.info("  NO data leakage: model has never seen 2024 data")
    logger.info("=" * 70)
    
    # ── Train temporal model ──
    logger.info("\n  Step 1: Train model on 2022-2023 only")
    temporal_model, feature_cols = train_temporal_model(
        features_dir, circuit_csv, train_seasons=[2022, 2023]
    )
    
    # Save temporal model for reference
    model_dir = Path("models")
    temporal_model.save_model(str(model_dir / "tyre_deg_temporal_2022_2023.json"))
    logger.info(f"  ✓ Temporal model saved")
    
    # ── Reconstruct actual 2024 strategies ──
    logger.info("\n  Step 2: Reconstruct 2024 race winner strategies")
    actual = reconstruct_actual_strategies(laps_dir, pitstops_path, circuit_csv, 2024)
    
    n_total = len(actual)
    n_wet = sum(1 for v in actual.values() if v["is_wet"])
    n_dry = n_total - n_wet
    logger.info(f"  {n_total} races ({n_dry} dry, {n_wet} wet)")
    
    # ── Run simulator with temporal model ──
    logger.info("\n  Step 3: Simulate 2024 strategies using 2022-2023 model")
    
    results_all = []
    results_dry = []
    stops_match_all = 0
    stops_match_dry = 0
    top3_all = 0
    top3_dry = 0
    top5_all = 0
    top5_dry = 0
    total_all = 0
    total_dry = 0
    
    for (s, rnd), real in sorted(actual.items()):
        try:
            circuit = load_circuit_config(
                real["circuit_key"], 2024, circuit_csv, sc_priors_path, weather_dir
            )
        except ValueError:
            continue
        
        strategies = generate_strategies(circuit)
        
        sim_results = []
        for i, strategy in enumerate(strategies):
            result = run_monte_carlo(
                strategy, circuit, temporal_model, feature_cols,
                fuel_config, n_sims=200, seed=42 + i,
            )
            sim_results.append(result)
        
        sim_results.sort(key=lambda x: x["median_time"])
        
        our_best = sim_results[0]
        our_stops = our_best["num_stops"]
        real_stops = real["n_stops"]
        
        matched = our_stops == real_stops
        in_top3 = any(r["num_stops"] == real_stops for r in sim_results[:3])
        in_top5 = any(r["num_stops"] == real_stops for r in sim_results[:5])
        
        total_all += 1
        if matched:
            stops_match_all += 1
        if in_top3:
            top3_all += 1
        if in_top5:
            top5_all += 1
        
        is_wet = real["is_wet"]
        
        if not is_wet:
            total_dry += 1
            if matched:
                stops_match_dry += 1
            if in_top3:
                top3_dry += 1
            if in_top5:
                top5_dry += 1
        
        marker = "✓" if matched else ("☁" if is_wet else "✗")
        
        entry = {
            "circuit": real["circuit_name"],
            "winner": real["winner"],
            "actual_stops": real_stops,
            "actual_compounds": real["compound_sequence"],
            "recommended": our_best["strategy_name"],
            "recommended_stops": our_stops,
            "stops_match": matched,
            "is_wet": is_wet,
            "in_top3": in_top3,
            "in_top5": in_top5,
        }
        results_all.append(entry)
        
        logger.info(
            f"  {marker} {real['circuit_name']:<28} "
            f"Actual: {real_stops}-stop | Ours: {our_stops}-stop"
            f"{'  [WET]' if is_wet else ''}"
        )
    
    # ── Summary ──
    logger.info("\n" + "═" * 70)
    logger.info("  VALIDATION RESULTS (Temporal Holdout)")
    logger.info("═" * 70)
    logger.info(f"  Training data:  2022-2023 seasons")
    logger.info(f"  Validation:     2024 season (unseen)")
    logger.info("")
    
    logger.info(f"  ALL RACES ({total_all}):")
    logger.info(f"    Exact stop match:   {stops_match_all}/{total_all} "
                f"({100*stops_match_all/max(total_all,1):.0f}%)")
    logger.info(f"    In top 3:           {top3_all}/{total_all} "
                f"({100*top3_all/max(total_all,1):.0f}%)")
    logger.info(f"    In top 5:           {top5_all}/{total_all} "
                f"({100*top5_all/max(total_all,1):.0f}%)")
    
    logger.info(f"\n  DRY RACES ONLY ({total_dry}):")
    logger.info(f"    Exact stop match:   {stops_match_dry}/{total_dry} "
                f"({100*stops_match_dry/max(total_dry,1):.0f}%)")
    logger.info(f"    In top 3:           {top3_dry}/{total_dry} "
                f"({100*top3_dry/max(total_dry,1):.0f}%)")
    logger.info(f"    In top 5:           {top5_dry}/{total_dry} "
                f"({100*top5_dry/max(total_dry,1):.0f}%)")
    
    wet_note = (
        "Wet races (marked ☁) are excluded from dry accuracy because "
        "the simulator only models dry compound strategies. Wet-weather "
        "strategy requires separate INTERMEDIATE/WET tyre models."
    )
    logger.info(f"\n  Note: {wet_note}")
    
    # ── Save ──
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output = {
        "methodology": {
            "type": "temporal_holdout",
            "train_seasons": [2022, 2023],
            "validation_season": 2024,
            "data_leakage": "none — model never sees 2024 data",
        },
        "results_all_races": {
            "total": total_all,
            "exact_match": stops_match_all,
            "exact_match_rate": round(stops_match_all / max(total_all, 1), 3),
            "top3_rate": round(top3_all / max(total_all, 1), 3),
            "top5_rate": round(top5_all / max(total_all, 1), 3),
        },
        "results_dry_only": {
            "total": total_dry,
            "exact_match": stops_match_dry,
            "exact_match_rate": round(stops_match_dry / max(total_dry, 1), 3),
            "top3_rate": round(top3_dry / max(total_dry, 1), 3),
            "top5_rate": round(top5_dry / max(total_dry, 1), 3),
        },
        "races": results_all,
        "note": wet_note,
    }
    
    with open(output_dir / "validation_proper_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  ✓ Results saved: results/validation_proper_report.json")


if __name__ == "__main__":
    run_proper_validation()
