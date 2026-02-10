"""
Pre-compute strategy results for all circuits -> export as TypeScript.

Usage:
    python -m src.scripts.precompute_all_strategies

Fixes from v1:
    - Different seed per strategy (seed=42+i) for variance
    - Deduplication: keeps only fastest variant per unique compound sequence
    - Logs 1-stop vs 2-stop counts to verify diversity
    - Re-runs all circuits (no cache) to ensure consistency
"""

import json
import re
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.simulation.strategy_simulator import (
    load_circuit_config,
    generate_strategies,
    run_monte_carlo,
)


def clean_strategy_name(raw: str) -> str:
    """Strip pit lap numbers and abbreviate compound names.
    '1-stop MEDIUM->HARD (25/32)' -> '1-stop M->H'
    """
    name = re.sub(r'\s*\(\d+(?:/\d+)*\)\s*', '', raw)
    name = name.replace('MEDIUM', 'M').replace('HARD', 'H').replace('SOFT', 'S')
    return name.strip()


def deduplicate_rankings(rankings: list) -> list:
    """Keep only the best (lowest median_time) variant of each unique
    clean strategy name. Re-rank and recalculate deltas."""
    seen = {}
    for r in rankings:
        key = clean_strategy_name(r["strategy_name"])
        if key not in seen or r["median_time"] < seen[key]["median_time"]:
            seen[key] = r

    deduped = sorted(seen.values(), key=lambda x: x["median_time"])

    if not deduped:
        return []

    best_median = deduped[0]["median_time"]
    for i, r in enumerate(deduped):
        r["rank"] = i + 1
        r["delta_to_best"] = round(r["median_time"] - best_median, 1)

    return deduped


def main():
    config = yaml.safe_load(open("configs/config.yaml"))

    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]

    model = xgb.XGBRegressor()
    model.load_model("models/tyre_deg_production.json")

    fuel_config = config["modeling"]["fuel_model"]

    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(config["paths"]["raw"]["fastf1"]) / "weather"

    circuits_df = pd.read_csv(circuit_csv)

    # Latest season per circuit
    latest = circuits_df.sort_values("season", ascending=False).drop_duplicates("circuit_key", keep="first")
    latest = latest.sort_values("round_number")

    logger.info("=" * 60)
    logger.info("  PRE-COMPUTING STRATEGIES FOR ALL CIRCUITS")
    logger.info(f"  {len(latest)} circuits to process")
    logger.info("=" * 60)

    all_results = {}
    n_sims = 1000
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for idx, (_, row) in enumerate(latest.iterrows()):
        circuit_key = row["circuit_key"]
        season = int(row["season"])
        circuit_name = row["circuit_name"]

        logger.info(f"\n[{idx+1}/{len(latest)}] {circuit_name} ({circuit_key}, {season})")

        try:
            t0 = time.time()

            circuit = load_circuit_config(
                circuit_key, season, circuit_csv, sc_priors_path, weather_dir,
            )

            strategies = generate_strategies(circuit)
            logger.info(f"  -> {len(strategies)} strategies, {n_sims} sims each")

            sim_results = []
            for i, strategy in enumerate(strategies):
                result = run_monte_carlo(
                    strategy, circuit, model, feature_cols,
                    fuel_config, n_sims=n_sims, seed=42 + i,
                )
                sim_results.append(result)

            sim_results.sort(key=lambda x: x["median_time"])
            elapsed = time.time() - t0

            # Deduplicate
            deduped = deduplicate_rankings(sim_results)

            output = {
                "circuit_key": circuit_key,
                "circuit_name": circuit_name,
                "season": season,
                "n_sims": n_sims,
                "total_strategies_raw": len(sim_results),
                "total_strategies_deduped": len(deduped),
                "elapsed_seconds": round(elapsed, 2),
                "rankings": deduped[:10],
            }

            result_file = results_dir / f"strategy_{circuit_key}_{season}.json"
            with open(result_file, "w") as f:
                json.dump(output, f, indent=2)

            all_results[f"{circuit_key}_{season}"] = output

            n_1stop = sum(1 for r in deduped if r["num_stops"] == 1)
            n_2stop = sum(1 for r in deduped if r["num_stops"] == 2)
            logger.info(f"  Done in {elapsed:.1f}s | Raw: {len(sim_results)} -> Deduped: {len(deduped)} ({n_1stop}x1-stop, {n_2stop}x2-stop)")
            logger.info(f"  Best: {clean_strategy_name(deduped[0]['strategy_name'])}")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    # -- Export TypeScript --
    logger.info("\n" + "=" * 60)
    logger.info("  EXPORTING TO TYPESCRIPT")
    logger.info("=" * 60)

    lines = []
    lines.append("export interface StrategyResult {")
    lines.append("  rank: number;")
    lines.append("  name: string;")
    lines.append("  cleanName: string;")
    lines.append("  compounds: string;")
    lines.append("  stops: number;")
    lines.append("  medianTime: number;")
    lines.append("  meanTime: number;")
    lines.append("  stdTime: number;")
    lines.append("  p5: number;")
    lines.append("  p95: number;")
    lines.append("  delta: number;")
    lines.append("  scEvents: number;")
    lines.append("}")
    lines.append("")
    lines.append("export interface CircuitStrategy {")
    lines.append("  circuit: string;")
    lines.append("  circuitName: string;")
    lines.append("  season: number;")
    lines.append("  nSims: number;")
    lines.append("  strategies: StrategyResult[];")
    lines.append("}")
    lines.append("")
    lines.append("export const strategyResults: Record<string, CircuitStrategy> = {")

    for key, data in sorted(all_results.items()):
        rankings = data["rankings"]
        if not rankings:
            continue

        best_median = rankings[0]["median_time"]
        cname = data["circuit_name"].replace('"', '\\"')

        lines.append(f'  "{key}": {{')
        lines.append(f'    circuit: "{data["circuit_key"]}",')
        lines.append(f'    circuitName: "{cname}",')
        lines.append(f'    season: {data["season"]},')
        lines.append(f'    nSims: {data["n_sims"]},')
        lines.append(f"    strategies: [")

        for i, r in enumerate(rankings):
            delta = round(r["median_time"] - best_median, 1)
            compound_seq = r.get("compound_sequence", r["strategy_name"])
            sc = r.get("mean_sc_events", 0)
            raw_name = r["strategy_name"].replace('"', '\\"')
            clean = clean_strategy_name(r["strategy_name"]).replace('"', '\\"')

            lines.append(
                f'      {{ rank: {i+1}, name: "{raw_name}", '
                f'cleanName: "{clean}", '
                f'compounds: "{compound_seq}", stops: {r["num_stops"]}, '
                f'medianTime: {r["median_time"]:.1f}, meanTime: {r["mean_time"]:.1f}, '
                f'stdTime: {r["std_time"]:.1f}, p5: {r["p5_time"]:.1f}, '
                f'p95: {r["p95_time"]:.1f}, delta: {delta}, '
                f'scEvents: {sc:.1f} }},'
            )

        lines.append("    ],")
        lines.append("  },")

    lines.append("};")
    lines.append("")

    # Validation data
    lines.append("export const validationData = {")
    lines.append("  folds: [")
    lines.append('    { label: "2022 -> 2023", trainStints: 967, cvMae: 0.1047, exactMatch: 40, top5Match: 50 },')
    lines.append('    { label: "2022-23 -> 2024", trainStints: 1982, cvMae: 0.0867, exactMatch: 52, top5Match: 71 },')
    lines.append('    { label: "2022-24 -> 2025", trainStints: 3006, cvMae: 0.0794, exactMatch: 71, top5Match: 86 },')
    lines.append("  ],")
    lines.append("  models: [")
    lines.append('    { name: "Ridge (Baseline)", mae: 0.0777, std: 0.013, time: 1.7 },')
    lines.append('    { name: "XGBoost (Primary)", mae: 0.0755, std: 0.012, time: 3.2 },')
    lines.append('    { name: "MLP (Neural Net)", mae: 0.0848, std: 0.015, time: 35.9 },')
    lines.append("  ],")
    lines.append("  shapFeatures: [")
    lines.append('    { name: "MeanHumidity", importance: 0.0134 },')
    lines.append('    { name: "MeanWindSpeed", importance: 0.0089 },')
    lines.append('    { name: "asphalt_grip", importance: 0.0078 },')
    lines.append('    { name: "TrackTempRange", importance: 0.0047 },')
    lines.append('    { name: "MeanTrackTemp", importance: 0.0046 },')
    lines.append('    { name: "traction_demand", importance: 0.0043 },')
    lines.append('    { name: "MeanAirTemp", importance: 0.0040 },')
    lines.append('    { name: "StintLength", importance: 0.0039 },')
    lines.append('    { name: "track_evolution", importance: 0.0016 },')
    lines.append('    { name: "StintNumber", importance: 0.0016 },')
    lines.append("  ],")
    lines.append("};")

    ts_content = "\n".join(lines) + "\n"

    frontend_path = Path("frontend/src/data/strategies.ts")
    frontend_path.parent.mkdir(parents=True, exist_ok=True)
    frontend_path.write_text(ts_content)
    logger.info(f"  Written to {frontend_path}")
    logger.info(f"  Total: {len(all_results)} circuits exported")


if __name__ == "__main__":
    main()
