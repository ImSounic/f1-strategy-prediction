"""
Rolling Temporal Validation (Expanding Window)
================================================
The proper way to validate time-dependent models:

    Fold 1: Train 2022         → Validate 2023
    Fold 2: Train 2022-2023    → Validate 2024
    Fold 3: Train 2022-2024    → Validate 2025

No data leakage. Shows how model improves with more historical data.

Output:
    results/validation_rolling_report.json

Usage:
    python -m src.analysis.strategy_validation_rolling
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
from sklearn.metrics import mean_absolute_error
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


def train_temporal_model(features_dir, circuit_csv, train_seasons):
    """Train XGBoost on specific seasons only."""
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

    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))

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

    return model, available, len(X), cv_mae


def reconstruct_actual_strategies(laps_dir, pitstops_path, circuits_csv, season):
    """Reconstruct what the race winner actually did."""
    results = pd.read_parquet(pitstops_path.parent / "results.parquet")
    pitstops = pd.read_parquet(pitstops_path)
    circuits = pd.read_csv(circuits_csv)

    actual = {}
    winners = results[(results["season"] == season) & (results["position"] == 1)]

    for _, winner in winners.iterrows():
        rnd = int(winner["round"])
        driver_id = winner["driverId"]
        driver_code = winner.get("driverCode", "UNK")

        pits = pitstops[
            (pitstops["season"] == season) &
            (pitstops["round"] == rnd) &
            (pitstops["driverId"] == driver_id)
        ].sort_values("lap")

        n_stops = len(pits)

        circuit_row = circuits[
            (circuits["season"] == season) & (circuits["round_number"] == rnd)
        ]
        if circuit_row.empty:
            continue

        circuit_name = circuit_row.iloc[0]["circuit_name"]
        circuit_key = circuit_row.iloc[0]["circuit_key"]

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
            "compounds": compounds,
            "compound_sequence": " → ".join(compounds) if compounds else "Unknown",
            "is_wet": is_wet,
        }

    return actual


def validate_fold(
    train_seasons, val_season, features_dir, circuit_csv,
    sc_priors_path, weather_dir, laps_dir, pitstops_path, fuel_config,
):
    """Run one fold of rolling validation."""
    logger.info(f"\n{'─' * 70}")
    logger.info(f"  FOLD: Train {train_seasons} → Validate {val_season}")
    logger.info(f"{'─' * 70}")

    # Train
    model, feature_cols, n_train, cv_mae = train_temporal_model(
        features_dir, circuit_csv, train_seasons,
    )
    logger.info(f"  Model: {n_train:,} training stints | CV MAE: {cv_mae:.4f}s")

    # Reconstruct actual
    actual = reconstruct_actual_strategies(laps_dir, pitstops_path, circuit_csv, val_season)
    n_wet = sum(1 for v in actual.values() if v["is_wet"])
    n_dry = len(actual) - n_wet
    logger.info(f"  Validation: {len(actual)} races ({n_dry} dry, {n_wet} wet)")

    # Simulate
    races = []
    counts = {"all": 0, "dry": 0}
    match = {"all": 0, "dry": 0}
    top3 = {"all": 0, "dry": 0}
    top5 = {"all": 0, "dry": 0}

    for (s, rnd), real in sorted(actual.items()):
        try:
            circuit = load_circuit_config(
                real["circuit_key"], val_season, circuit_csv, sc_priors_path, weather_dir,
            )
        except ValueError:
            continue

        strategies = generate_strategies(circuit)

        sim_results = []
        for i, strat in enumerate(strategies):
            result = run_monte_carlo(
                strat, circuit, model, feature_cols,
                fuel_config, n_sims=200, seed=42 + i,
            )
            sim_results.append(result)

        sim_results.sort(key=lambda x: x["median_time"])

        our_stops = sim_results[0]["num_stops"]
        real_stops = real["n_stops"]
        matched = our_stops == real_stops
        in_top3 = any(r["num_stops"] == real_stops for r in sim_results[:3])
        in_top5 = any(r["num_stops"] == real_stops for r in sim_results[:5])
        is_wet = real["is_wet"]

        counts["all"] += 1
        if matched: match["all"] += 1
        if in_top3: top3["all"] += 1
        if in_top5: top5["all"] += 1

        if not is_wet:
            counts["dry"] += 1
            if matched: match["dry"] += 1
            if in_top3: top3["dry"] += 1
            if in_top5: top5["dry"] += 1

        marker = "✓" if matched else ("☁" if is_wet else "✗")
        logger.info(
            f"  {marker} {real['circuit_name']:<28} "
            f"Actual: {real_stops}-stop ({real['compound_sequence'][:35]:<35}) | "
            f"Ours: {our_stops}-stop{'  [WET]' if is_wet else ''}"
        )

        races.append({
            "circuit": real["circuit_name"],
            "winner": real["winner"],
            "actual_stops": real_stops,
            "actual_compounds": real["compound_sequence"],
            "recommended_stops": our_stops,
            "recommended_strategy": sim_results[0]["strategy_name"],
            "stops_match": matched,
            "in_top3": in_top3,
            "in_top5": in_top5,
            "is_wet": is_wet,
        })

    def rate(n, d):
        return round(n / max(d, 1), 3)

    fold_result = {
        "train_seasons": train_seasons,
        "val_season": val_season,
        "n_training_stints": n_train,
        "cv_mae": round(cv_mae, 4),
        "all_races": {
            "total": counts["all"],
            "exact_match": match["all"],
            "exact_rate": rate(match["all"], counts["all"]),
            "top3_rate": rate(top3["all"], counts["all"]),
            "top5_rate": rate(top5["all"], counts["all"]),
        },
        "dry_races": {
            "total": counts["dry"],
            "exact_match": match["dry"],
            "exact_rate": rate(match["dry"], counts["dry"]),
            "top3_rate": rate(top3["dry"], counts["dry"]),
            "top5_rate": rate(top5["dry"], counts["dry"]),
        },
        "races": races,
    }

    logger.info(f"\n  Summary — All: {match['all']}/{counts['all']} "
                f"({100*rate(match['all'], counts['all']):.0f}%) | "
                f"Dry: {match['dry']}/{counts['dry']} "
                f"({100*rate(match['dry'], counts['dry']):.0f}%)")

    return fold_result


def run_rolling_validation(config_path: str = "configs/config.yaml"):
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
    logger.info("  ROLLING TEMPORAL VALIDATION (Expanding Window)")
    logger.info("=" * 70)
    logger.info("  Fold 1: Train 2022         → Validate 2023")
    logger.info("  Fold 2: Train 2022-2023    → Validate 2024")
    logger.info("  Fold 3: Train 2022-2024    → Validate 2025")

    folds = [
        ([2022], 2023),
        ([2022, 2023], 2024),
        ([2022, 2023, 2024], 2025),
    ]

    t0 = time.time()
    all_folds = []

    for train_seasons, val_season in folds:
        fold = validate_fold(
            train_seasons, val_season, features_dir, circuit_csv,
            sc_priors_path, weather_dir, laps_dir, pitstops_path, fuel_config,
        )
        all_folds.append(fold)

    elapsed = time.time() - t0

    # ── Final summary table ──
    logger.info("\n" + "═" * 70)
    logger.info("  ROLLING VALIDATION SUMMARY")
    logger.info("═" * 70)
    logger.info(f"  {'Fold':<25} {'Train':>8} {'CV MAE':>8} {'All':>10} {'Dry':>10} {'Dry Top5':>10}")
    logger.info(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    total_match_dry = 0
    total_dry = 0
    total_top5_dry = 0

    for fold in all_folds:
        ts = fold["train_seasons"]
        vs = fold["val_season"]
        label = f"{'→'.join(str(s) for s in ts)} → {vs}"
        n_train = fold["n_training_stints"]
        mae = fold["cv_mae"]

        a = fold["all_races"]
        d = fold["dry_races"]

        total_match_dry += d["exact_match"]
        total_dry += d["total"]
        total_top5_dry += int(d["top5_rate"] * d["total"])

        logger.info(
            f"  {label:<25} {n_train:>7,} {mae:>7.4f}s "
            f"{a['exact_match']}/{a['total']:>2} ({100*a['exact_rate']:>3.0f}%) "
            f"{d['exact_match']}/{d['total']:>2} ({100*d['exact_rate']:>3.0f}%) "
            f"({100*d['top5_rate']:>3.0f}%)"
        )

    avg_dry_rate = total_match_dry / max(total_dry, 1)
    avg_top5_rate = total_top5_dry / max(total_dry, 1)

    logger.info(f"\n  AGGREGATE (dry races across all folds):")
    logger.info(f"    Exact match: {total_match_dry}/{total_dry} ({100*avg_dry_rate:.0f}%)")
    logger.info(f"    Top 5 match: {total_top5_dry}/{total_dry} ({100*avg_top5_rate:.0f}%)")
    logger.info(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = {
        "methodology": {
            "type": "rolling_temporal_validation",
            "description": "Expanding window: each fold adds one more season to training",
            "folds": [
                {"train": [2022], "validate": 2023},
                {"train": [2022, 2023], "validate": 2024},
                {"train": [2022, 2023, 2024], "validate": 2025},
            ],
            "data_leakage": "none",
        },
        "aggregate_dry": {
            "exact_match": total_match_dry,
            "total": total_dry,
            "exact_rate": round(avg_dry_rate, 3),
            "top5_rate": round(avg_top5_rate, 3),
        },
        "folds": all_folds,
    }

    with open(output_dir / "validation_rolling_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  ✓ Results saved: results/validation_rolling_report.json")


if __name__ == "__main__":
    run_rolling_validation()
