"""
Position-aware strategy selection — validation
==============================================
For each dry race, evaluate candidate strategies for the WINNER (rest of field
on real strategies) and pick by two objectives from the same sim runs:
  - time-optimal     = min expected in-race time
  - position-optimal = min expected finishing position
Each pick is scored against the winner's actual strategy (stop-count + full
compound sequence). Reports whether position-awareness predicts reality better.

Output: results/position_strategy_report.json

Usage:
    python -m src.analysis.position_strategy_validation --seasons 2022 2023 2024 2025 --n-sims 15
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy, generate_common_strategies
from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params
from src.simulation.regulation_profiles import get_profile
from src.analysis.position_validation import load_config, reconstruct_field
from src.analysis.position_strategy import dedupe_by_sequence, argmin_by
from src.analysis.strategy_match import score_race

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def evaluate_candidates(circuit, grid_drivers, strategies, target_idx,
                        candidates, n_sims, base_seed, profile):
    stats = []
    for cand in candidates:
        tsum, psum = 0.0, 0
        for i in range(n_sims):
            sim = MultiCarRaceSim(
                circuit=circuit, drivers=grid_drivers, strategies=strategies,
                target_driver_idx=target_idx, target_strategy=cand,
                greedy_sc=False, profile=profile,
            )
            r = sim.run(seed=base_seed + i)
            tsum += r["target_time"]
            psum += r["target_position"]
        stats.append({
            "seq": list(cand.compound_sequence),
            "num_stops": cand.num_stops,
            "mean_time": tsum / n_sims,
            "mean_pos": psum / n_sims,
        })
    return stats


def _score_pick(pick, real):
    """stop-count + full-sequence match of a pick vs the actual strategy."""
    sim_results = [{"compound_sequence": pick["seq"], "num_stops": pick["num_stops"]}]
    s = score_race(sim_results, real, top_ks=(3,))
    return bool(s["stop_match"]), bool(s["strategy_exact"])


def evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile):
    code_to_driver = {d.code: d for d in drivers}
    field = [f for f in field if f["code"] in code_to_driver]
    winners = [f for f in field if f["finish"] == 1]
    if not winners or len(field) < 2:
        return None
    winner = winners[0]

    field = sorted(field, key=lambda f: f["grid"])
    grid_drivers = [code_to_driver[f["code"]] for f in field]
    strategies = [Strategy(stints=f["stints"], name=f["code"]) for f in field]
    target_idx = next(i for i, f in enumerate(field) if f["code"] == winner["code"])

    candidates = dedupe_by_sequence(generate_common_strategies(circuit.total_laps))
    if not candidates:
        return None

    stats = evaluate_candidates(circuit, grid_drivers, strategies, target_idx,
                                candidates, n_sims, 2000, profile)
    time_pick = argmin_by(stats, "mean_time")
    pos_pick = argmin_by(stats, "mean_pos")

    real = {"compounds": [n for n, _ in winner["stints"]],
            "n_stops": len(winner["stints"]) - 1}
    t_stop, t_exact = _score_pick(time_pick, real)
    p_stop, p_exact = _score_pick(pos_pick, real)

    return {
        "season": season, "round": int(rnd), "winner": winner["code"],
        "actual_seq": real["compounds"], "actual_stops": real["n_stops"],
        "time_pick_seq": time_pick["seq"], "pos_pick_seq": pos_pick["seq"],
        "time_stop_match": t_stop, "time_strat_exact": t_exact,
        "pos_stop_match": p_stop, "pos_strat_exact": p_exact,
    }


def _rates(races):
    n = len(races)
    if n == 0:
        return {}
    def rate(k):
        return round(sum(1 for r in races if r[k]) / n, 3)
    return {
        "n_races": n,
        "time_stop_rate": rate("time_stop_match"),
        "time_strat_rate": rate("time_strat_exact"),
        "pos_stop_rate": rate("pos_stop_match"),
        "pos_strat_rate": rate("pos_strat_exact"),
    }


def run(seasons, n_sims, config_path="configs/config.yaml"):
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

    all_races = []
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
            row = evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile)
            if row is None:
                continue
            row["circuit"] = ckey
            races.append(row)
            logger.info(f"  {season} r{rnd:>2} {ckey:<14} "
                        f"time[stop={row['time_stop_match']:d} exact={row['time_strat_exact']:d}] "
                        f"pos[stop={row['pos_stop_match']:d} exact={row['pos_strat_exact']:d}]")
        season_reports.append({"season": season, **_rates(races), "races": races})
        all_races.extend(races)
        logger.info(f"=== {season}: {_rates(races)}")

    overall = _rates(all_races)
    report = {
        "methodology": {
            "description": "Per dry race, evaluate deduped candidate strategies for the "
                           "winner (field on real strategies); pick by min expected time vs "
                           "min expected finishing position from the same sim runs; score each "
                           "pick vs the winner's actual strategy.",
            "n_sims": n_sims,
            "caveat": "time objective uses in-race target time (proxy); both picks share the "
                      "same candidate set, equally capped by candidate coverage.",
        },
        "overall": overall,
        "seasons": season_reports,
    }
    out = Path("results/position_strategy_report.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nOVERALL {overall}")
    logger.info(f"Saved: {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Position-aware strategy selection validation")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--n-sims", type=int, default=15)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run(args.seasons, args.n_sims, args.config)


if __name__ == "__main__":
    main()
