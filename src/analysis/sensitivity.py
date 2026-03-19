"""
Sensitivity Analysis
=====================
Tests how strategy rankings change when key simulation parameters are perturbed.

For each parameter, runs the MC simulator at ±10%, ±20% of the default value
and records:
  - Whether the top strategy changes
  - How many rank swaps occur in the top 5
  - Time delta between strategies at each perturbation level

Parameters tested (highest impact on strategy rankings):
  1. Pit loss seconds (per circuit, ±20%)
  2. SC probability per race (±20%)
  3. Tyre degradation quadratic coefficient (0.002 ±50%)
  4. Tyre degradation exponent (1.3 ±15%)
  5. Fuel effect per kg (0.033 ±30%)
  6. SC pace factor (1.40 ±7%)
  7. SC pit loss reduction (0.50 ±20%)
  8. Lap time noise std (0.3 ±50%)

Output:
    frontend/src/data/sensitivityResults.ts
    results/sensitivity_report.json

Usage:
    python -m src.analysis.sensitivity
    python -m src.analysis.sensitivity --circuit bahrain --n-sims 200
    python -m src.analysis.sensitivity --circuits all --n-sims 200
"""

import argparse
import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.simulation.strategy_simulator import (
    CircuitConfig,
    Strategy,
    StintPlan,
    generate_strategies,
    precompute_deg_rates,
    load_circuit_config,
    COMPOUND_HARDNESS,
)


# ═══════════════════════════════════════════════════════════════════
#  PARAMETER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SensitivityParam:
    """A parameter to test sensitivity for."""
    key: str
    label: str
    default: float
    unit: str
    source: str  # citation or justification
    perturbations: list[float]  # multipliers, e.g. [0.8, 0.9, 1.0, 1.1, 1.2]
    category: str  # 'pit', 'tyre', 'safety_car', 'fuel', 'noise'


PARAMS = [
    SensitivityParam(
        key="pit_loss",
        label="Pit Loss Time",
        default=0.0,  # circuit-specific, set per run
        unit="s",
        source="Per-circuit from Pirelli data (pit lane length × speed limit)",
        perturbations=[0.80, 0.90, 1.00, 1.10, 1.20],
        category="pit",
    ),
    SensitivityParam(
        key="sc_probability",
        label="Safety Car Probability",
        default=0.0,  # circuit-specific
        unit="",
        source="Bayesian posterior from historical SC events per circuit (models/safety_car_priors.json)",
        perturbations=[0.80, 0.90, 1.00, 1.10, 1.20],
        category="safety_car",
    ),
    SensitivityParam(
        key="deg_quadratic",
        label="Tyre Wear Curve Steepness",
        default=0.002,
        unit="s/lap²",
        source="Fitted constant — controls cliff severity in tyre_deg = rate×age + 0.002×age^1.3",
        perturbations=[0.50, 0.75, 1.00, 1.25, 1.50],
        category="tyre",
    ),
    SensitivityParam(
        key="deg_exponent",
        label="Tyre Wear Exponent",
        default=1.3,
        unit="",
        source="Fitted constant — controls non-linearity of tyre cliff (1.0=linear, 2.0=quadratic)",
        perturbations=[0.85, 0.92, 1.00, 1.08, 1.15],
        category="tyre",
    ),
    SensitivityParam(
        key="fuel_effect",
        label="Fuel Weight Penalty",
        default=0.033,
        unit="s/kg",
        source="F1 consensus ~0.03-0.04s per kg (FIA technical regs, Scarbs F1 analysis)",
        perturbations=[0.70, 0.85, 1.00, 1.15, 1.30],
        category="fuel",
    ),
    SensitivityParam(
        key="sc_pace_factor",
        label="SC Pace Multiplier",
        default=1.40,
        unit="×",
        source="SC lap times ~40% slower than race pace (FIA regulations, delta time system)",
        perturbations=[0.93, 0.96, 1.00, 1.04, 1.07],
        category="safety_car",
    ),
    SensitivityParam(
        key="sc_pit_reduction",
        label="SC Pit Time Saving",
        default=0.50,
        unit="×",
        source="Under SC, pit loss reduced ~50% due to compressed field (varies by pit lane length)",
        perturbations=[0.80, 0.90, 1.00, 1.10, 1.20],
        category="pit",
    ),
    SensitivityParam(
        key="lap_noise",
        label="Lap Time Variability",
        default=0.3,
        unit="s",
        source="Std dev of lap-to-lap noise from traffic, driver variation, track evolution",
        perturbations=[0.50, 0.75, 1.00, 1.25, 1.50],
        category="noise",
    ),
]


# ═══════════════════════════════════════════════════════════════════
#  MODIFIED SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_race_with_params(
    strategy: Strategy,
    circuit: CircuitConfig,
    precomputed_deg_rates: list,
    fuel_config: dict,
    sc_prob_per_lap: float,
    vsc_prob_per_lap: float,
    base_pace: float,
    burn_rate: float,
    start_fuel: float,
    fuel_effect: float,
    rng: np.random.Generator,
    # Sensitivity overrides
    deg_quadratic: float = 0.002,
    deg_exponent: float = 1.3,
    sc_pace_factor: float = 1.40,
    sc_pit_reduction: float = 0.50,
    lap_noise_std: float = 0.3,
    pit_loss_override: float | None = None,
) -> tuple:
    """Simulation with parameterized constants for sensitivity testing."""
    total_laps = circuit.total_laps
    pit_loss = pit_loss_override if pit_loss_override is not None else circuit.pit_loss_seconds
    total_time = 0.0
    sc_count = 0
    sc_remaining = 0
    vsc_remaining = 0
    current_stint = 0
    laps_in_stint = 0

    for lap in range(1, total_laps + 1):
        stint = strategy.stints[current_stint]
        laps_in_stint += 1

        fuel_remaining = max(0, start_fuel - burn_rate * (lap - 1))
        fuel_time = fuel_remaining * fuel_effect

        deg_rate = precomputed_deg_rates[current_stint] + rng.normal(0, 0.01)
        deg_rate = max(0.005, deg_rate)
        tyre_deg = deg_rate * laps_in_stint + deg_quadratic * (laps_in_stint ** deg_exponent)

        if sc_remaining > 0:
            lap_time = base_pace * sc_pace_factor
            sc_remaining -= 1
        elif vsc_remaining > 0:
            lap_time = base_pace * 1.20
            vsc_remaining -= 1
        else:
            lap_time = base_pace + fuel_time + tyre_deg + rng.normal(0, lap_noise_std)

            if rng.random() < sc_prob_per_lap and 1 < lap < total_laps - 3:
                sc_remaining = rng.integers(3, 7)
                lap_time = base_pace * sc_pace_factor
                sc_count += 1
            elif rng.random() < vsc_prob_per_lap and 1 < lap < total_laps - 2:
                vsc_remaining = rng.integers(2, 5)
                lap_time = base_pace * 1.20

        # Pit stop
        if current_stint < len(strategy.stints) - 1 and laps_in_stint >= stint.target_laps:
            if sc_remaining > 0:
                lap_time += pit_loss * sc_pit_reduction + rng.normal(0, 0.5)
            elif vsc_remaining > 0:
                lap_time += pit_loss * 0.65 + rng.normal(0, 0.5)
            else:
                lap_time += pit_loss + rng.normal(0, 0.5)
            current_stint += 1
            laps_in_stint = 0

        total_time += lap_time

    return total_time, sc_count


def run_mc_with_params(
    strategy, circuit, deg_rates, fuel_config, n_sims, seed,
    **param_overrides,
) -> dict:
    """Run MC simulation with parameter overrides."""
    rng = np.random.default_rng(seed)

    start_fuel = fuel_config["start_fuel_kg"]
    fuel_effect = param_overrides.get("fuel_effect", fuel_config["fuel_effect_per_kg_seconds"])
    burn_rate = start_fuel / circuit.total_laps
    base_pace = 90.0

    sc_prob = param_overrides.get("sc_probability", circuit.sc_prob_per_race)
    if sc_prob > 0:
        sc_prob_per_lap = 1 - (1 - sc_prob) ** (1 / circuit.total_laps)
    else:
        sc_prob_per_lap = 0
    vsc_prob = circuit.vsc_prob_per_race
    if vsc_prob > 0:
        vsc_prob_per_lap = 1 - (1 - vsc_prob) ** (1 / circuit.total_laps)
    else:
        vsc_prob_per_lap = 0

    times = []
    for _ in range(n_sims):
        t, _ = simulate_race_with_params(
            strategy, circuit, deg_rates, fuel_config,
            sc_prob_per_lap, vsc_prob_per_lap, base_pace,
            burn_rate, start_fuel, fuel_effect, rng,
            deg_quadratic=param_overrides.get("deg_quadratic", 0.002),
            deg_exponent=param_overrides.get("deg_exponent", 1.3),
            sc_pace_factor=param_overrides.get("sc_pace_factor", 1.40),
            sc_pit_reduction=param_overrides.get("sc_pit_reduction", 0.50),
            lap_noise_std=param_overrides.get("lap_noise", 0.3),
            pit_loss_override=param_overrides.get("pit_loss", None),
        )
        times.append(t)

    arr = np.array(times)
    return {
        "strategy_name": strategy.name,
        "compound_sequence": strategy.compound_sequence,
        "num_stops": strategy.num_stops,
        "median_time": float(np.median(arr)),
    }


# ═══════════════════════════════════════════════════════════════════
#  SENSITIVITY SWEEP
# ═══════════════════════════════════════════════════════════════════

def run_sensitivity_for_circuit(
    circuit_key: str,
    season: int,
    n_sims: int,
    config: dict,
) -> dict:
    """Run sensitivity analysis for one circuit."""
    raw_paths = config["paths"]["raw"]
    fuel_config = config["modeling"]["fuel_model"]

    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"

    circuit = load_circuit_config(circuit_key, season, circuit_csv, sc_priors_path, weather_dir)

    # Load model
    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    with open("models/comparison_results.json") as f:
        feature_cols = json.load(f)["experiment"]["feature_columns"]

    strategies = generate_strategies(circuit)

    # Precompute deg rates for all strategies
    all_deg_rates = []
    for strat in strategies:
        rates = precompute_deg_rates(strat, circuit, deg_model, feature_cols)
        all_deg_rates.append(rates)

    logger.info(f"  {circuit.circuit_name}: {len(strategies)} strategies × {len(PARAMS)} params × 5 levels")

    # Baseline: run all strategies with default params
    baseline_results = []
    for i, (strat, rates) in enumerate(zip(strategies, all_deg_rates)):
        r = run_mc_with_params(strat, circuit, rates, fuel_config, n_sims, seed=42 + i)
        baseline_results.append(r)
    baseline_results.sort(key=lambda x: x["median_time"])
    baseline_top = baseline_results[0]["strategy_name"]
    baseline_top5 = [r["strategy_name"] for r in baseline_results[:5]]

    # Set circuit-specific defaults
    PARAMS[0].default = circuit.pit_loss_seconds
    PARAMS[1].default = circuit.sc_prob_per_race

    param_results = []

    for param in PARAMS:
        levels = []

        for mult in param.perturbations:
            actual_value = param.default * mult

            # Build overrides
            overrides = {}
            if param.key == "pit_loss":
                overrides["pit_loss"] = actual_value
            elif param.key == "sc_probability":
                overrides["sc_probability"] = min(actual_value, 0.99)
            elif param.key == "deg_quadratic":
                overrides["deg_quadratic"] = actual_value
            elif param.key == "deg_exponent":
                overrides["deg_exponent"] = actual_value
            elif param.key == "fuel_effect":
                overrides["fuel_effect"] = actual_value
            elif param.key == "sc_pace_factor":
                overrides["sc_pace_factor"] = actual_value
            elif param.key == "sc_pit_reduction":
                overrides["sc_pit_reduction"] = actual_value
            elif param.key == "lap_noise":
                overrides["lap_noise"] = actual_value

            # Run all strategies with this parameter level
            results = []
            for i, (strat, rates) in enumerate(zip(strategies, all_deg_rates)):
                r = run_mc_with_params(strat, circuit, rates, fuel_config, n_sims, seed=42 + i, **overrides)
                results.append(r)

            results.sort(key=lambda x: x["median_time"])
            top_strategy = results[0]["strategy_name"]
            top5 = [r["strategy_name"] for r in results[:5]]

            # Compute rank changes
            top_changed = top_strategy != baseline_top
            rank_swaps = sum(1 for s in baseline_top5 if s not in top5)

            levels.append({
                "multiplier": mult,
                "value": round(actual_value, 4),
                "label": f"{mult:.0%}" if mult != 1.0 else "Default",
                "topStrategy": top_strategy,
                "topChanged": top_changed,
                "top5RankSwaps": rank_swaps,
                "top5": [{"rank": j + 1, "name": r["strategy_name"], "median": round(r["median_time"], 1)} for j, r in enumerate(results[:5])],
            })

        param_results.append({
            "key": param.key,
            "label": param.label,
            "default": round(param.default, 4),
            "unit": param.unit,
            "source": param.source,
            "category": param.category,
            "levels": levels,
        })

    return {
        "circuitKey": circuit_key,
        "circuitName": circuit.circuit_name,
        "season": season,
        "nSims": n_sims,
        "baselineTop": baseline_top,
        "baselineTop5": baseline_top5,
        "parameters": param_results,
    }


# ═══════════════════════════════════════════════════════════════════
#  TYPESCRIPT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_typescript(all_results: dict):
    """Generate frontend data file."""
    ts_path = Path("frontend/src/data/sensitivityResults.ts")

    lines = [
        "// Auto-generated by src/analysis/sensitivity.py — do not edit manually",
        "",
        "export interface SensitivityLevel {",
        "  multiplier: number;",
        "  value: number;",
        "  label: string;",
        "  topStrategy: string;",
        "  topChanged: boolean;",
        "  top5RankSwaps: number;",
        "  top5: { rank: number; name: string; median: number }[];",
        "}",
        "",
        "export interface SensitivityParam {",
        "  key: string;",
        "  label: string;",
        "  default: number;",
        "  unit: string;",
        "  source: string;",
        "  category: string;",
        "  levels: SensitivityLevel[];",
        "}",
        "",
        "export interface CircuitSensitivity {",
        "  circuitKey: string;",
        "  circuitName: string;",
        "  season: number;",
        "  nSims: number;",
        "  baselineTop: string;",
        "  baselineTop5: string[];",
        "  parameters: SensitivityParam[];",
        "}",
        "",
        f"export const sensitivityData: Record<string, CircuitSensitivity> = {json.dumps(all_results, indent=2)};",
        "",
    ]

    ts_path.write_text("\n".join(lines))
    logger.info(f"  ✓ TypeScript saved: {ts_path}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for strategy simulation parameters.")
    parser.add_argument("--circuit", type=str, default=None, help="Single circuit key")
    parser.add_argument("--circuits", type=str, default=None, help="Comma-separated circuit keys or 'all'")
    parser.add_argument("--n-sims", type=int, default=200, help="MC simulations per strategy per level")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    # Determine circuits
    if args.circuit:
        circuit_keys = [args.circuit]
    elif args.circuits == "all":
        results_dir = Path("results")
        circuit_keys = sorted(
            p.stem.replace("strategy_", "").replace(f"_{args.season}", "")
            for p in results_dir.glob(f"strategy_*_{args.season}.json")
        )
    elif args.circuits:
        circuit_keys = [c.strip() for c in args.circuits.split(",")]
    else:
        circuit_keys = ["bahrain", "monaco", "monza", "silverstone", "singapore"]

    logger.info("=" * 65)
    logger.info("  SENSITIVITY ANALYSIS")
    logger.info("=" * 65)
    logger.info(f"  Circuits: {len(circuit_keys)}")
    logger.info(f"  Parameters: {len(PARAMS)}")
    logger.info(f"  Simulations per level: {args.n_sims}")

    t0 = time.time()
    all_results = {}

    for ck in circuit_keys:
        try:
            result = run_sensitivity_for_circuit(ck, args.season, args.n_sims, config)
            all_results[ck] = result
        except Exception as e:
            logger.error(f"  Failed {ck}: {e}")

    elapsed = time.time() - t0
    logger.info(f"\n  Total time: {elapsed:.1f}s")

    # Save
    generate_typescript(all_results)

    output_dir = Path("results")
    with open(output_dir / "sensitivity_report.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  ✓ JSON saved: results/sensitivity_report.json")


if __name__ == "__main__":
    main()
