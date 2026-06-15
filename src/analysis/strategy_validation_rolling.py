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
from src.simulation.compound_prior import CompoundPrior
from src.analysis.strategy_match import score_race


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
    """Reconstruct what the race winner actually did.

    Uses stint_features.parquet as primary source for compound data
    (always available), falling back to raw laps parquet if needed.
    """
    results = pd.read_parquet(pitstops_path.parent / "results.parquet")
    pitstops = pd.read_parquet(pitstops_path)
    circuits = pd.read_csv(circuits_csv)

    # Load stint features for compound data (more reliable than raw laps)
    features_dir = Path("data/features")
    stints_df = None
    if (features_dir / "stint_features.parquet").exists():
        stints_df = pd.read_parquet(features_dir / "stint_features.parquet")

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
        hard_c = circuit_row.iloc[0]["hard_compound"]
        med_c = circuit_row.iloc[0]["medium_compound"]
        soft_c = circuit_row.iloc[0]["soft_compound"]

        compounds = []

        # Primary: use stint features (always available)
        if stints_df is not None:
            driver_stints = stints_df[
                (stints_df["Season"] == season) &
                (stints_df["RoundNumber"] == rnd) &
                (stints_df["Driver"] == driver_code)
            ].sort_values("StintNumber")

            if not driver_stints.empty:
                for _, stint in driver_stints.iterrows():
                    c = stint["Compound"]
                    # Map C1-C6 to compound names
                    if c == soft_c:
                        compounds.append("SOFT")
                    elif c == med_c:
                        compounds.append("MEDIUM")
                    elif c == hard_c:
                        compounds.append("HARD")
                    else:
                        compounds.append(c)  # INTERMEDIATE, WET, etc.

        # Fallback: raw laps parquet
        if not compounds:
            laps_files = list(laps_dir.glob(f"{season}_{rnd:02d}_*_R.parquet"))
            if not laps_files:
                laps_files = list(laps_dir.glob(f"{season}_{rnd}_*_R.parquet"))

            if laps_files:
                laps_df = pd.read_parquet(laps_files[0])
                wlaps = laps_df[laps_df["Driver"] == driver_code]
                if not wlaps.empty and "Stint" in wlaps.columns:
                    compounds = wlaps.groupby("Stint")["Compound"].first().tolist()

        is_wet = any(c in ["INTERMEDIATE", "WET"] for c in compounds)

        # Heuristic: if we see far fewer stints than stops+1, compounds
        # were likely INTERMEDIATE/WET that got filtered during feature engineering
        if not is_wet and n_stops >= 2 and len(compounds) <= n_stops - 1:
            is_wet = True

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
    compound_prior: CompoundPrior | None = None,
    prior_blend: float = 0.3,
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
    stop_match = {"all": 0, "dry": 0}    # right NUMBER of stops (headline "exact")
    strat_exact = {"all": 0, "dry": 0}   # exact compound SEQUENCE (top-1)
    strat_top3 = {"all": 0, "dry": 0}    # actual sequence within top-3 distinct
    strat_top5 = {"all": 0, "dry": 0}    # actual sequence within top-5 distinct

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

        # Rerank with compound prior if available
        if compound_prior is not None:
            sim_results = compound_prior.rerank_strategies(
                sim_results, real["circuit_key"], blend_weight=prior_blend,
            )

        # Score: stop-count match AND full-strategy match (top-1/3/5 over
        # DISTINCT compound sequences). See src/analysis/strategy_match.py.
        sc = score_race(sim_results, real, top_ks=(3, 5))
        is_wet = real["is_wet"]

        for scope in (["all"] if is_wet else ["all", "dry"]):
            counts[scope] += 1
            if sc["stop_match"]:     stop_match[scope] += 1
            if sc["strategy_exact"]: strat_exact[scope] += 1
            if sc["strategy_top3"]:  strat_top3[scope] += 1
            if sc["strategy_top5"]:  strat_top5[scope] += 1

        marker = "✓" if sc["stop_match"] else ("☁" if is_wet else "✗")
        logger.info(
            f"  {marker} {real['circuit_name']:<28} "
            f"Actual: {real['n_stops']}-stop ({real['compound_sequence'][:35]:<35}) | "
            f"Ours: {sc['recommended_stops']}-stop{'  [WET]' if is_wet else ''}"
        )

        races.append({
            "circuit": real["circuit_name"],
            "winner": real["winner"],
            "actual_stops": real["n_stops"],
            "actual_compounds": real["compound_sequence"],
            "recommended_stops": sc["recommended_stops"],
            "recommended_strategy": sim_results[0]["strategy_name"],
            "recommended_sequence": sc["recommended_sequence"],
            "stops_match": sc["stop_match"],
            "strategy_exact": sc["strategy_exact"],
            "in_top3": sc["strategy_top3"],
            "in_top5": sc["strategy_top5"],
            "is_wet": is_wet,
        })

    def rate(n, d):
        return round(n / max(d, 1), 3)

    def block(scope):
        # exact_* = stop-count match (headline). top3/top5 = full-strategy match
        # over distinct compound sequences. strategy_exact_* = full top-1.
        return {
            "total": counts[scope],
            "exact_match": stop_match[scope],
            "exact_rate": rate(stop_match[scope], counts[scope]),
            "strategy_exact_match": strat_exact[scope],
            "strategy_exact_rate": rate(strat_exact[scope], counts[scope]),
            "top3_match": strat_top3[scope],
            "top3_rate": rate(strat_top3[scope], counts[scope]),
            "top5_match": strat_top5[scope],
            "top5_rate": rate(strat_top5[scope], counts[scope]),
        }

    fold_result = {
        "train_seasons": train_seasons,
        "val_season": val_season,
        "n_training_stints": n_train,
        "cv_mae": round(cv_mae, 4),
        "all_races": block("all"),
        "dry_races": block("dry"),
        "races": races,
    }

    logger.info(f"\n  Summary — All: {stop_match['all']}/{counts['all']} "
                f"({100*rate(stop_match['all'], counts['all']):.0f}%) | "
                f"Dry: {stop_match['dry']}/{counts['dry']} "
                f"({100*rate(stop_match['dry'], counts['dry']):.0f}%)")

    return fold_result


def _build_temporal_prior(features_dir, circuit_csv, results_path, train_seasons):
    """Build compound prior from training seasons only (no data leakage)."""
    from src.simulation.compound_prior import CompoundPrior
    from collections import Counter, defaultdict

    features_dir = Path(features_dir)
    stints = pd.read_parquet(features_dir / "stint_features.parquet")
    circuits = pd.read_csv(circuit_csv)
    results = pd.read_parquet(results_path)

    # Filter to training seasons ONLY
    stints = stints[stints["Season"].isin(train_seasons)]
    results = results[results["season"].isin(train_seasons)]

    merged = stints.merge(
        circuits[["season", "round_number", "circuit_key",
                   "hard_compound", "medium_compound", "soft_compound",
                   "tyre_stress"]],
        left_on=["Season", "RoundNumber"],
        right_on=["season", "round_number"],
        how="inner",
    )

    def _map(row):
        c = row["Compound"]
        if c == row["soft_compound"]: return "SOFT"
        elif c == row["medium_compound"]: return "MEDIUM"
        elif c == row["hard_compound"]: return "HARD"
        return c

    merged["CompoundName"] = merged.apply(_map, axis=1)
    dry = merged[merged["CompoundName"].isin(["SOFT", "MEDIUM", "HARD"])]

    # Circuit categories
    circuit_stress = circuits.groupby("circuit_key")["tyre_stress"].mean().to_dict()
    def _cat(ck):
        s = circuit_stress.get(ck, 50)
        return "low_stress" if s < 35 else "high_stress" if s >= 55 else "med_stress"
    circuit_cats = {ck: _cat(ck) for ck in circuit_stress}

    # Build sequences
    sequences = (
        dry.sort_values(["Season", "RoundNumber", "Driver", "StintNumber"])
        .groupby(["Season", "RoundNumber", "Driver", "circuit_key"])
        .agg(
            Strategy=("CompoundName", lambda x: "-".join(x)),
            FirstCompound=("CompoundName", "first"),
            NumStints=("CompoundName", "count"),
        )
        .reset_index()
    )
    sequences["NumStops"] = sequences["NumStints"] - 1
    sequences["Category"] = sequences["circuit_key"].map(circuit_cats)

    # Starting compound priors
    start_probs = {}
    for cat in ["low_stress", "med_stress", "high_stress"]:
        subset = sequences[sequences["Category"] == cat]
        counts = subset["FirstCompound"].value_counts()
        total = counts.sum()
        start_probs[cat] = {c: counts.get(c, 0) / max(total, 1) for c in ["SOFT", "MEDIUM", "HARD"]}

    # Transitions
    transitions = defaultdict(Counter)
    for (_, _, driver), group in dry.sort_values(
        ["Season", "RoundNumber", "Driver", "StintNumber"]
    ).groupby(["Season", "RoundNumber", "Driver"]):
        compounds = group.sort_values("StintNumber")["CompoundName"].tolist()
        for i in range(len(compounds) - 1):
            transitions[compounds[i]][compounds[i + 1]] += 1

    transition_probs = {}
    for from_c, to_counts in transitions.items():
        total = sum(to_counts.values())
        transition_probs[from_c] = {to_c: count / total for to_c, count in to_counts.items()}

    # Stop count distribution
    stop_probs = {}
    for cat in ["low_stress", "med_stress", "high_stress"]:
        subset = sequences[sequences["Category"] == cat]
        counts = subset["NumStops"].value_counts()
        total = counts.sum()
        stop_probs[cat] = {int(s): c / max(total, 1) for s, c in counts.items()}

    # Per-circuit strategy counts
    circuit_strat_counts = {}
    for ck, group in sequences.groupby("circuit_key"):
        if group["Season"].nunique() >= 1:
            circuit_strat_counts[ck] = Counter(group["Strategy"].tolist())

    return CompoundPrior(
        start_probs=start_probs,
        transition_probs=transition_probs,
        stop_probs=stop_probs,
        circuit_strategy_counts=circuit_strat_counts,
        circuit_categories=circuit_cats,
    )


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
    logger.info("  ROLLING TEMPORAL VALIDATION (Expanding Window + Compound Prior)")
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
        # Build compound prior from training seasons only (no data leakage)
        compound_prior = _build_temporal_prior(
            features_dir, circuit_csv, pitstops_path.parent / "results.parquet",
            train_seasons,
        )
        logger.info(f"  Compound prior built from seasons {train_seasons}")

        fold = validate_fold(
            train_seasons, val_season, features_dir, circuit_csv,
            sc_priors_path, weather_dir, laps_dir, pitstops_path, fuel_config,
            compound_prior=compound_prior, prior_blend=0.15,
        )
        all_folds.append(fold)

    elapsed = time.time() - t0

    # ── Final summary table ──
    logger.info("\n" + "═" * 70)
    logger.info("  ROLLING VALIDATION SUMMARY")
    logger.info("═" * 70)
    logger.info(f"  {'Fold':<25} {'Train':>8} {'CV MAE':>8} {'All':>10} {'Dry':>10} {'Dry Top5':>10}")
    logger.info(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    total_stop_dry = 0
    total_dry = 0
    total_strat_exact_dry = 0
    total_strat_top3_dry = 0
    total_strat_top5_dry = 0

    for fold in all_folds:
        ts = fold["train_seasons"]
        vs = fold["val_season"]
        label = f"{'→'.join(str(s) for s in ts)} → {vs}"
        n_train = fold["n_training_stints"]
        mae = fold["cv_mae"]

        a = fold["all_races"]
        d = fold["dry_races"]

        # Sum raw integer counts (the old code rebuilt counts from rounded
        # rates and truncated with int(), which made top5 < exact — impossible).
        total_stop_dry += d["exact_match"]
        total_dry += d["total"]
        total_strat_exact_dry += d["strategy_exact_match"]
        total_strat_top3_dry += d["top3_match"]
        total_strat_top5_dry += d["top5_match"]

        logger.info(
            f"  {label:<25} {n_train:>7,} {mae:>7.4f}s "
            f"{a['exact_match']}/{a['total']:>2} ({100*a['exact_rate']:>3.0f}%) "
            f"{d['exact_match']}/{d['total']:>2} ({100*d['exact_rate']:>3.0f}%) "
            f"({100*d['top5_rate']:>3.0f}%)"
        )

    def agg_rate(n):
        return round(n / max(total_dry, 1), 3)

    logger.info(f"\n  AGGREGATE (dry races across all folds):")
    logger.info(f"    Stop-count exact: {total_stop_dry}/{total_dry} ({100*agg_rate(total_stop_dry):.0f}%)")
    logger.info(f"    Strategy exact:   {total_strat_exact_dry}/{total_dry} ({100*agg_rate(total_strat_exact_dry):.0f}%)")
    logger.info(f"    Strategy top-5:   {total_strat_top5_dry}/{total_dry} ({100*agg_rate(total_strat_top5_dry):.0f}%)")
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
            "total": total_dry,
            "exact_match": total_stop_dry,
            "exact_rate": agg_rate(total_stop_dry),
            "strategy_exact_match": total_strat_exact_dry,
            "strategy_exact_rate": agg_rate(total_strat_exact_dry),
            "top3_match": total_strat_top3_dry,
            "top3_rate": agg_rate(total_strat_top3_dry),
            "top5_match": total_strat_top5_dry,
            "top5_rate": agg_rate(total_strat_top5_dry),
        },
        "folds": all_folds,
    }

    with open(output_dir / "validation_rolling_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  ✓ Results saved: results/validation_rolling_report.json")


if __name__ == "__main__":
    run_rolling_validation()
