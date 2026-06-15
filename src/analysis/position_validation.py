"""
Position validation harness
===========================
Feeds each race's real grid + every car's real strategy into MultiCarRaceSim
and scores predicted vs actual finishing order (Spearman + position MAE), over
classified finishers, for dry races across the requested seasons.

Output: results/position_validation_report.json

Usage:
    python -m src.analysis.position_validation --seasons 2022 2023 2024 2025 --n-sims 30
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy
from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params
from src.simulation.regulation_profiles import get_profile
from src.analysis.position_match import score_positions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Stint laps naturally fall ~15-20% short of race distance (lap 1, pit in/out,
# and cleaned/SC laps are excluded from stint_features), so dry races cluster
# ~0.73-0.95 coverage while genuinely wet races (INTER/WET laps dropped) sit
# below ~0.66. 0.70 separates them cleanly without discarding valid dry races.
DRY_COVERAGE_MIN = 0.70


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def reconstruct_field(season, rnd, stints_df, results_df, circuits_df):
    """Return classified-finisher field for a race, or None if wet/insufficient.

    Each entry: {code, grid, finish, stints=[(name, laps), ...]}.
    """
    crow = circuits_df[(circuits_df["season"] == season) &
                       (circuits_df["round_number"] == rnd)]
    if crow.empty:
        return None
    crow = crow.iloc[0]
    total_laps = int(crow["total_laps"])
    cmap = {crow["soft_compound"]: "SOFT",
            crow["medium_compound"]: "MEDIUM",
            crow["hard_compound"]: "HARD"}

    res = results_df[(results_df["season"] == season) & (results_df["round"] == rnd)]
    res = res[res["position"].notna()]
    if res.empty:
        return None

    field = []
    winner_cover = 0.0
    for _, r in res.iterrows():
        code = r["driverCode"]
        ds = stints_df[(stints_df["Season"] == season) &
                       (stints_df["RoundNumber"] == rnd) &
                       (stints_df["Driver"] == code)].sort_values("StintNumber")
        if ds.empty:
            continue
        stints = [(cmap.get(s["Compound"], "MEDIUM"), int(s["StintLength"]))
                  for _, s in ds.iterrows()]
        cover = sum(n for _, n in stints) / max(total_laps, 1)
        if int(r["position"]) == 1:
            winner_cover = cover
        grid = int(r["grid"]) if r["grid"] > 0 else 20
        field.append({"code": code, "grid": grid, "finish": int(r["position"]), "stints": stints})

    if len(field) < 2:
        return None
    if winner_cover and winner_cover < DRY_COVERAGE_MIN:
        return None  # likely wet (stint laps don't cover the race)
    return field


def simulate_order(circuit, drivers, field, n_sims, base_seed, profile):
    """Run the field on fixed real strategies; return {code: predicted rank}."""
    code_to_driver = {d.code: d for d in drivers}
    field = [f for f in field if f["code"] in code_to_driver]
    if len(field) < 2:
        return None

    field = sorted(field, key=lambda f: f["grid"])
    grid_drivers = [code_to_driver[f["code"]] for f in field]
    strategies = [Strategy(stints=f["stints"], name=f["code"]) for f in field]

    sums = {f["code"]: 0 for f in field}
    for i in range(n_sims):
        sim = MultiCarRaceSim(
            circuit=circuit,
            drivers=grid_drivers,
            strategies=strategies,
            target_driver_idx=0,
            target_strategy=strategies[0],
            greedy_sc=False,          # everyone follows their fixed real strategy
            profile=profile,
        )
        result = sim.run(seed=base_seed + i)
        fin = result["finishing_positions"]   # 1-indexed, per grid index
        for idx, f in enumerate(field):
            sums[f["code"]] += fin[idx]

    mean_pos = {c: s / n_sims for c, s in sums.items()}
    ordered = sorted(mean_pos, key=lambda c: mean_pos[c])
    return {c: rank + 1 for rank, c in enumerate(ordered)}


def run_validation(seasons, n_sims, config_path="configs/config.yaml"):
    config = load_config(config_path)
    raw = config["paths"]["raw"]
    circuit_csv = Path(raw["supplementary"]) / "pirelli_circuit_characteristics.csv"
    circuits_df = pd.read_csv(circuit_csv)
    results_df = pd.read_parquet(Path(raw["jolpica"]) / "results.parquet")
    stints_df = pd.read_parquet(Path("data/features") / "stint_features.parquet")

    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    with open("models/comparison_results.json") as f:
        feature_cols = json.load(f)["experiment"]["feature_columns"]

    season_reports = []
    for season in seasons:
        profile = get_profile(season)
        drivers, _teams, overtaking = load_drivers(f"configs/drivers_{season}.json")
        rounds = sorted(results_df[results_df["season"] == season]["round"].unique())
        races = []
        for rnd in rounds:
            field = reconstruct_field(season, rnd, stints_df, results_df, circuits_df)
            if field is None:
                continue
            ckey = circuits_df[(circuits_df["season"] == season) &
                               (circuits_df["round_number"] == rnd)].iloc[0]["circuit_key"]
            try:
                circuit = load_circuit_as_params(ckey, season, config, overtaking,
                                                 deg_model, feature_cols)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  {season} r{rnd} ({ckey}): circuit load failed: {e}")
                continue
            predicted = simulate_order(circuit, drivers, field, n_sims, 1000, profile)
            if predicted is None:
                continue
            actual = {f["code"]: f["finish"] for f in field if f["code"] in predicted}
            score = score_positions(predicted, actual)
            score.update({"season": season, "round": int(rnd), "circuit": ckey})
            races.append(score)
            sp = score["spearman"]
            logger.info(f"  {season} r{rnd:>2} {ckey:<14} "
                        f"spearman={sp:.3f} MAE={score['position_mae']:.2f} n={score['n']}"
                        if sp is not None else
                        f"  {season} r{rnd:>2} {ckey:<14} (insufficient)")

        valid = [r for r in races if r["spearman"] is not None]
        mean_sp = float(np.mean([r["spearman"] for r in valid])) if valid else None
        pooled_mae = (float(np.average([r["position_mae"] for r in valid],
                                       weights=[r["n"] for r in valid])) if valid else None)
        season_reports.append({
            "season": season, "n_races": len(valid),
            "mean_spearman": mean_sp, "pooled_position_mae": pooled_mae,
            "races": races,
        })
        logger.info(f"=== {season}: races={len(valid)} mean_spearman={mean_sp} "
                    f"pooled_MAE={pooled_mae}")

    all_valid = [r for s in season_reports for r in s["races"] if r["spearman"] is not None]
    overall = {
        "n_races": len(all_valid),
        "mean_spearman": float(np.mean([r["spearman"] for r in all_valid])) if all_valid else None,
        "pooled_position_mae": (float(np.average([r["position_mae"] for r in all_valid],
                                                 weights=[r["n"] for r in all_valid]))
                                if all_valid else None),
    }
    report = {
        "methodology": {
            "description": "Real grid + real per-car strategies fed to MultiCarRaceSim; "
                           "predicted vs actual finishing order over classified finishers, dry races.",
            "n_sims": n_sims, "dry_coverage_min": DRY_COVERAGE_MIN,
            "limitations": ["no DNF/retirement model (finishers only)",
                            "wet races skipped via stint coverage",
                            "grid from results.grid; season-aggregate driver form"],
        },
        "overall": overall,
        "seasons": season_reports,
    }
    out = Path("results/position_validation_report.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nOverall: races={overall['n_races']} "
                f"mean_spearman={overall['mean_spearman']} "
                f"pooled_MAE={overall['pooled_position_mae']}")
    logger.info(f"Saved: {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Position validation harness")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--n-sims", type=int, default=30)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_validation(args.seasons, args.n_sims, args.config)


if __name__ == "__main__":
    main()
