"""
Pre-compute strategy results for all circuits → export as TypeScript.

Usage:
    python -m src.scripts.precompute_all_strategies

Outputs:
    frontend/src/data/strategies.ts  (overwritten with all circuit data)
    results/strategy_*.json          (individual circuit results)
"""

import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.simulation.strategy_simulator import (
    load_circuit_config,
    generate_strategies,
    run_monte_carlo,
)


def main():
    config = yaml.safe_load(open("configs/config.yaml"))

    # Load model
    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]

    model = xgb.XGBRegressor()
    model.load_model("models/tyre_deg_production.json")

    fuel_config = config["modeling"]["fuel_model"]

    # Load circuits
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(config["paths"]["raw"]["fastf1"]) / "weather"

    circuits_df = pd.read_csv(circuit_csv)

    # Use latest season for each circuit
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

        # Check if already computed
        result_file = results_dir / f"strategy_{circuit_key}_{season}.json"
        if result_file.exists():
            logger.info(f"  → Loading cached results from {result_file}")
            with open(result_file) as f:
                data = json.load(f)
            all_results[f"{circuit_key}_{season}"] = data
            continue

        try:
            t0 = time.time()

            circuit = load_circuit_config(
                circuit_key, season, circuit_csv, sc_priors_path, weather_dir,
            )

            strategies = generate_strategies(circuit)
            logger.info(f"  → {len(strategies)} strategies, {n_sims} sims each")

            sim_results = []
            for strategy in strategies:
                result = run_monte_carlo(
                    strategy, circuit, model, feature_cols,
                    fuel_config, n_sims=n_sims, seed=42,
                )
                sim_results.append(result)

            sim_results.sort(key=lambda x: x["median_time"])
            elapsed = time.time() - t0

            # Save JSON
            output = {
                "circuit_key": circuit_key,
                "circuit_name": circuit_name,
                "season": season,
                "n_sims": n_sims,
                "total_strategies": len(sim_results),
                "elapsed_seconds": round(elapsed, 2),
                "rankings": sim_results,
            }

            with open(result_file, "w") as f:
                json.dump(output, f, indent=2)

            all_results[f"{circuit_key}_{season}"] = output
            total_sims = len(strategies) * n_sims
            logger.info(f"  ✓ {total_sims:,} sims in {elapsed:.1f}s — best: {sim_results[0]['strategy_name']}")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            continue

    # ── Export as TypeScript ──
    logger.info("\n" + "=" * 60)
    logger.info("  EXPORTING TO TYPESCRIPT")
    logger.info("=" * 60)

    ts_lines = []
    ts_lines.append("export interface StrategyResult {")
    ts_lines.append("  rank: number;")
    ts_lines.append("  name: string;")
    ts_lines.append("  compounds: string;")
    ts_lines.append("  stops: number;")
    ts_lines.append("  medianTime: number;")
    ts_lines.append("  meanTime: number;")
    ts_lines.append("  stdTime: number;")
    ts_lines.append("  p5: number;")
    ts_lines.append("  p95: number;")
    ts_lines.append("  delta: number;")
    ts_lines.append("  scEvents: number;")
    ts_lines.append("}")
    ts_lines.append("")
    ts_lines.append("export interface CircuitStrategy {")
    ts_lines.append("  circuit: string;")
    ts_lines.append("  circuitName: string;")
    ts_lines.append("  season: number;")
    ts_lines.append("  nSims: number;")
    ts_lines.append("  strategies: StrategyResult[];")
    ts_lines.append("}")
    ts_lines.append("")
    ts_lines.append("export const strategyResults: Record<string, CircuitStrategy> = {")

    for key, data in sorted(all_results.items()):
        rankings = data["rankings"][:10]  # Top 10 only
        best_median = rankings[0]["median_time"]

        ts_lines.append(f'  {key}: {{')
        ts_lines.append(f'    circuit: "{data["circuit_key"]}",')
        ts_lines.append(f'    circuitName: "{data["circuit_name"]}",')
        ts_lines.append(f'    season: {data["season"]},')
        ts_lines.append(f'    nSims: {data["n_sims"]},')
        ts_lines.append(f'    strategies: [')

        for i, r in enumerate(rankings):
            delta = round(r["median_time"] - best_median, 1)
            compound_seq = r.get("compound_sequence", r["strategy_name"])
            sc = r.get("mean_sc_events", 0)

            ts_lines.append(f'      {{ rank: {i+1}, name: "{r["strategy_name"]}", '
                          f'compounds: "{compound_seq}", stops: {r["num_stops"]}, '
                          f'medianTime: {r["median_time"]:.1f}, meanTime: {r["mean_time"]:.1f}, '
                          f'stdTime: {r["std_time"]:.1f}, p5: {r["p5_time"]:.1f}, '
                          f'p95: {r["p95_time"]:.1f}, delta: {delta}, '
                          f'scEvents: {sc:.1f} }},')

        ts_lines.append(f'    ],')
        ts_lines.append(f'  }},')

    ts_lines.append("};")
    ts_lines.append("")

    # Validation data (unchanged)
    ts_lines.append("export const validationData = {")
    ts_lines.append("  folds: [")
    ts_lines.append('    { label: "2022 → 2023", trainStints: 967, cvMae: 0.1047, exactMatch: 40, top5Match: 50 },')
    ts_lines.append('    { label: "2022-23 → 2024", trainStints: 1982, cvMae: 0.0867, exactMatch: 52, top5Match: 71 },')
    ts_lines.append('    { label: "2022-24 → 2025", trainStints: 3006, cvMae: 0.0794, exactMatch: 71, top5Match: 86 },')
    ts_lines.append("  ],")
    ts_lines.append("  models: [")
    ts_lines.append('    { name: "Ridge (Baseline)", mae: 0.0777, std: 0.013, time: 1.7 },')
    ts_lines.append('    { name: "XGBoost (Primary)", mae: 0.0755, std: 0.012, time: 3.2 },')
    ts_lines.append('    { name: "MLP (Neural Net)", mae: 0.0848, std: 0.015, time: 35.9 },')
    ts_lines.append("  ],")
    ts_lines.append("  shapFeatures: [")
    ts_lines.append('    { name: "MeanHumidity", importance: 0.0134 },')
    ts_lines.append('    { name: "MeanWindSpeed", importance: 0.0089 },')
    ts_lines.append('    { name: "asphalt_grip", importance: 0.0078 },')
    ts_lines.append('    { name: "TrackTempRange", importance: 0.0047 },')
    ts_lines.append('    { name: "MeanTrackTemp", importance: 0.0046 },')
    ts_lines.append('    { name: "traction_demand", importance: 0.0043 },')
    ts_lines.append('    { name: "MeanAirTemp", importance: 0.0040 },')
    ts_lines.append('    { name: "StintLength", importance: 0.0039 },')
    ts_lines.append('    { name: "track_evolution", importance: 0.0016 },')
    ts_lines.append('    { name: "StintNumber", importance: 0.0016 },')
    ts_lines.append("  ],")
    ts_lines.append("};")

    ts_content = "\n".join(ts_lines) + "\n"

    # Write to frontend
    frontend_path = Path("frontend/src/data/strategies.ts")
    frontend_path.parent.mkdir(parents=True, exist_ok=True)
    frontend_path.write_text(ts_content)
    logger.info(f"  ✓ Written to {frontend_path}")

    # ── Also export circuits.ts from actual CSV data ──
    logger.info("\n  Exporting circuits.ts...")

    with open(sc_priors_path) as f:
        sc_priors = json.load(f)

    # Country code mapping
    country_map = {
        "bahrain": "BH", "jeddah": "SA", "albert_park": "AU", "suzuka": "JP",
        "shanghai": "CN", "miami": "US", "imola": "IT", "monaco": "MC",
        "montreal": "CA", "barcelona": "ES", "spielberg": "AT", "silverstone": "GB",
        "hungaroring": "HU", "spa": "BE", "zandvoort": "NL", "monza": "IT",
        "baku": "AZ", "singapore": "SG", "cota": "US", "mexico": "MX",
        "interlagos": "BR", "las_vegas": "US", "lusail": "QA", "yas_marina": "AE",
        "paul_ricard": "FR",
    }

    char_cols = [
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution",
    ]

    ct_lines = []
    ct_lines.append("export interface Circuit {")
    ct_lines.append("  key: string;")
    ct_lines.append("  name: string;")
    ct_lines.append("  country: string;")
    ct_lines.append("  totalLaps: number;")
    ct_lines.append("  pitLoss: number;")
    ct_lines.append("  scProbability: number;")
    ct_lines.append("  compounds: string;")
    ct_lines.append("  characteristics: Record<string, number>;")
    ct_lines.append("}")
    ct_lines.append("")
    ct_lines.append("export const circuits: Circuit[] = [")

    for _, row in latest.iterrows():
        ck = row["circuit_key"]
        sc = sc_priors.get(ck, {})
        cc = country_map.get(ck, "??")
        compounds = f"{row['hard_compound']}/{row['medium_compound']}/{row['soft_compound']}"

        chars = ", ".join(
            f'{col.replace("asphalt_abrasiveness","abrasiveness").replace("asphalt_grip","grip").replace("traction_demand","traction").replace("braking_severity","braking").replace("lateral_forces","lateral").replace("tyre_stress","stress").replace("downforce_level","downforce").replace("track_evolution","evolution")}: {int(row[col])}'
            for col in char_cols
        )

        ct_lines.append(f'  {{')
        ct_lines.append(f'    key: "{ck}", name: "{row["circuit_name"]}", country: "{cc}",')
        ct_lines.append(f'    totalLaps: {int(row["total_laps"])}, pitLoss: {row["pit_loss_seconds"]}, scProbability: {sc.get("bayesian_sc_prob", 0.55):.3f}, compounds: "{compounds}",')
        ct_lines.append(f'    characteristics: {{ {chars} }},')
        ct_lines.append(f'  }},')

    ct_lines.append("];")

    circuits_ts_path = Path("frontend/src/data/circuits.ts")
    circuits_ts_path.write_text("\n".join(ct_lines) + "\n")
    logger.info(f"  ✓ Written to {circuits_ts_path}")

    # Summary
    logger.info(f"\n  Total: {len(all_results)} circuits pre-computed")
    logger.info(f"  Results saved to results/strategy_*.json")
    logger.info(f"  TypeScript exported to {frontend_path}")
    logger.info(f"  Circuits exported to {circuits_ts_path}")


if __name__ == "__main__":
    main()
