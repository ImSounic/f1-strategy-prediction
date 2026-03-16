"""
Pre-Compute Race Strategy Scenarios
=====================================
Evaluates optimal strategies for all driver × circuit × grid position combinations.

For each combo:
  1. Build 20-car grid with target driver at specified position
  2. Assign AI drivers their MC-optimal strategy
  3. MC search: test top candidate strategies × N sims
  4. Score by median finishing position
  5. Store: best strategy, position stats, sample race

Output: JSON per circuit + combined TypeScript file for frontend.

Usage:
    python -m src.simulation.precompute_scenarios [--circuits all] [--n-sims 50]
"""

import json
import logging
import time
import yaml
import numpy as np
from pathlib import Path
from itertools import product

from src.simulation.multi_car_sim import (
    MultiCarRaceSim, DriverConfig, CircuitParams, Strategy,
    generate_common_strategies, build_grid, find_target_in_grid,
    COMPOUND_DEG_BASE,
)
from src.simulation.strategy_simulator import (
    load_circuit_config,
    predict_degradation,
    COMPOUND_HARDNESS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_drivers(path: str = "configs/drivers_2024.json") -> tuple:
    """Load driver database. Returns (drivers_list, teams_dict, overtaking_dict)."""
    with open(path) as f:
        data = json.load(f)
    
    drivers = []
    for d in data["drivers"]:
        drivers.append(DriverConfig(
            code=d["code"],
            name=d["name"],
            team=d["team"],
            pace_delta=d["pace_delta"],
            overtaking=d["overtaking"],
            tyre_management=d["tyre_management"],
            teammate_code=d.get("teammate", ""),
        ))
    
    return drivers, data["teams"], data.get("circuit_overtaking_difficulty", {})


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_available_circuits(season: int = 2025) -> list:
    """Find all circuits with precomputed MC results."""
    results_dir = Path("results")
    keys = []
    for f in sorted(results_dir.glob(f"strategy_*_{season}.json")):
        key = f.stem.replace("strategy_", "").replace(f"_{season}", "")
        keys.append(key)
    return keys


def load_circuit_as_params(
    circuit_key: str,
    season: int,
    config: dict,
    overtaking_difficulty: dict,
    deg_model=None,
    feature_cols=None,
) -> CircuitParams:
    """
    Load circuit using the existing strategy_simulator loader,
    then convert to our CircuitParams format.
    """
    raw_paths = config["paths"]["raw"]
    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"

    # Load using existing loader → returns CircuitConfig
    cc = load_circuit_config(circuit_key, season, circuit_csv, sc_priors_path, weather_dir)

    # Compute deg rates from XGBoost if model available
    deg_rates = {}
    if deg_model is not None and feature_cols is not None:
        for compound_name in ["SOFT", "MEDIUM", "HARD"]:
            actual = getattr(cc, f"{compound_name.lower()}_compound", compound_name)
            hardness = COMPOUND_HARDNESS.get(actual, 3)
            rate = predict_degradation(
                deg_model, hardness, 1, 25, cc, feature_cols,
            )
            deg_rates[compound_name] = float(max(0.005, min(rate, 0.5)))
    else:
        deg_rates = COMPOUND_DEG_BASE.copy()

    return CircuitParams(
        circuit_key=circuit_key,
        circuit_name=cc.circuit_name,
        total_laps=cc.total_laps,
        pit_loss_seconds=cc.pit_loss_seconds,
        sc_prob_per_race=cc.sc_prob_per_race,
        vsc_prob_per_race=getattr(cc, "vsc_prob_per_race", 0.15),
        overtaking_difficulty=overtaking_difficulty.get(circuit_key, 0.5),
        deg_rates=deg_rates,
    )


def get_ai_strategy(circuit: CircuitParams) -> Strategy:
    """
    Get a reasonable default strategy for AI drivers.
    Uses the most common 1-stop medium→hard strategy.
    """
    total = circuit.total_laps
    pit_lap = int(total * 0.50)
    return Strategy(
        stints=[("MEDIUM", pit_lap), ("HARD", total - pit_lap)],
        name="MH default",
    )


def _generate_narrative(result: dict, grid_position: int, total_laps: int) -> str:
    """Generate a short narrative describing what happened in a race."""
    pos = result["target_position"]
    sc_laps = result.get("sc_laps", [])
    pit_events = [e for e in result.get("pit_events", []) if e["driver_idx"] == result.get("_target_idx", -1)]
    positions = result["target_history"]["positions"]
    
    gain = grid_position - pos
    n_sc = len(set(sc_laps))
    
    parts = []
    
    # Opening: result
    if gain > 0:
        parts.append(f"Gained {gain} positions (P{grid_position} → P{pos}).")
    elif gain < 0:
        parts.append(f"Lost {abs(gain)} positions (P{grid_position} → P{pos}).")
    else:
        parts.append(f"Held position P{pos}.")
    
    # SC description
    if n_sc == 0:
        parts.append("Clean race — no safety car interventions.")
    else:
        # Find SC timing
        unique_sc = sorted(set(sc_laps))
        first_sc = unique_sc[0]
        race_third = total_laps / 3
        
        if first_sc <= race_third:
            timing = "early"
        elif first_sc <= 2 * race_third:
            timing = "mid-race"
        else:
            timing = "late"
        
        if n_sc == 1:
            parts.append(f"Safety car deployed {timing} (lap {first_sc}).")
        else:
            parts.append(f"{n_sc} safety car periods — first on lap {first_sc} ({timing}).")
        
        # Did SC help or hurt?
        if len(positions) > 0:
            # Position before first SC vs a few laps after
            pre_sc_idx = max(0, first_sc - 2)
            post_sc_idx = min(len(positions) - 1, first_sc + 5)
            if pre_sc_idx < len(positions) and post_sc_idx < len(positions):
                pre_pos = positions[pre_sc_idx]
                post_pos = positions[post_sc_idx]
                if post_pos < pre_pos:
                    parts.append(f"SC helped — gained {pre_pos - post_pos} positions during neutralisation.")
                elif post_pos > pre_pos:
                    parts.append(f"SC hurt — lost {post_pos - pre_pos} positions as field compressed.")
    
    # Pit strategy
    pit_laps = result["target_history"].get("pit_laps", [])
    if pit_laps:
        parts.append(f"Pitted on lap{'s' if len(pit_laps) > 1 else ''} {', '.join(str(l) for l in pit_laps)}.")
    
    return " ".join(parts)


def evaluate_strategy(
    circuit: CircuitParams,
    drivers: list[DriverConfig],
    target_code: str,
    grid_position: int,
    strategy: Strategy,
    ai_strategy: Strategy,
    n_sims: int = 50,
    base_seed: int = 1000,
) -> dict:
    """
    Evaluate a single strategy for the target driver.
    Runs n_sims races and returns position statistics + 5 representative races.
    """
    # Build grid
    target_driver_idx = next(i for i, d in enumerate(drivers) if d.code == target_code)
    grid = build_grid(drivers, target_driver_idx, grid_position)
    target_in_grid = find_target_in_grid(grid, target_code)
    
    # AI strategies (same for all AI drivers)
    strategies = [ai_strategy] * len(grid)
    
    positions = []
    all_results = []
    
    for i in range(n_sims):
        seed = base_seed + i
        sim = MultiCarRaceSim(
            circuit=circuit,
            drivers=grid,
            strategies=strategies,
            target_driver_idx=target_in_grid,
            target_strategy=strategy,
            greedy_sc=True,
        )
        result = sim.run(seed=seed)
        result["_target_idx"] = target_in_grid
        positions.append(result["target_position"])
        all_results.append(result)
    
    positions_arr = np.array(positions)
    
    # ── Pick 5 representative races ──
    
    def _extract_sample(r):
        """Extract compact sample race data from a full sim result."""
        return {
            "positions": r["target_history"]["positions"],
            "tyre_ages": r["target_history"]["tyre_ages"],
            "compounds": r["target_history"]["compounds"],
            "pit_laps": r["target_history"]["pit_laps"],
            "sc_laps": r["sc_laps"],
            "target_position": r["target_position"],
            "narrative": _generate_narrative(r, grid_position, circuit.total_laps),
        }
    
    # Sort by finishing position
    sorted_by_pos = sorted(range(n_sims), key=lambda i: (positions[i], i))
    
    best_idx = sorted_by_pos[0]
    worst_idx = sorted_by_pos[-1]
    median_idx = sorted_by_pos[len(sorted_by_pos) // 2]
    
    sample_races = {
        "best": _extract_sample(all_results[best_idx]),
        "worst": _extract_sample(all_results[worst_idx]),
        "median": _extract_sample(all_results[median_idx]),
    }
    
    # Early SC: SC in first third, pick one closest to median position
    early_sc_candidates = []
    late_sc_candidates = []
    third = circuit.total_laps / 3
    
    for i, r in enumerate(all_results):
        sc_set = sorted(set(r["sc_laps"]))
        if not sc_set:
            continue
        first_sc = sc_set[0]
        if first_sc <= third:
            early_sc_candidates.append(i)
        if first_sc >= 2 * third:
            late_sc_candidates.append(i)
    
    median_pos = float(np.median(positions_arr))
    
    if early_sc_candidates:
        # Pick the early SC race closest to median outcome
        best_early = min(early_sc_candidates, key=lambda i: abs(positions[i] - median_pos))
        sample_races["early_sc"] = _extract_sample(all_results[best_early])
    
    if late_sc_candidates:
        best_late = min(late_sc_candidates, key=lambda i: abs(positions[i] - median_pos))
        sample_races["late_sc"] = _extract_sample(all_results[best_late])
    
    return {
        "strategy_name": strategy.name,
        "compound_sequence": strategy.compound_sequence,
        "pit_laps": strategy.pit_laps,
        "num_stops": strategy.num_stops,
        "median_pos": float(np.median(positions_arr)),
        "mean_pos": float(np.mean(positions_arr)),
        "p5_pos": float(np.percentile(positions_arr, 5)),
        "p95_pos": float(np.percentile(positions_arr, 95)),
        "pos_distribution": {int(p): int(np.sum(positions_arr == p)) for p in range(1, 21)},
        "sample_races": sample_races,
    }


def optimize_scenario(
    circuit: CircuitParams,
    drivers: list[DriverConfig],
    target_code: str,
    grid_position: int,
    n_sims: int = 50,
    n_strategies: int = 8,
) -> dict:
    """
    Find optimal strategy for a driver × circuit × grid position.
    Tests top candidate strategies and picks best by median position.
    """
    # Generate candidate strategies
    candidates = generate_common_strategies(circuit.total_laps)
    ai_strategy = get_ai_strategy(circuit)
    
    # Quick eval: run each with fewer sims to narrow down
    quick_results = []
    for s in candidates[:20]:  # limit candidates for speed
        r = evaluate_strategy(
            circuit, drivers, target_code, grid_position,
            s, ai_strategy, n_sims=max(10, n_sims // 5),
        )
        quick_results.append(r)
    
    # Sort by median position and take top N
    quick_results.sort(key=lambda r: r["median_pos"])
    top_strategies = quick_results[:n_strategies]
    
    # Full eval on top strategies
    full_results = []
    for qr in top_strategies:
        s = Strategy(
            stints=list(zip(qr["compound_sequence"], 
                           [qr["pit_laps"][0] if qr["pit_laps"] else circuit.total_laps] + 
                           [qr["pit_laps"][i] - qr["pit_laps"][i-1] if i > 0 else 0 
                            for i in range(1, len(qr["pit_laps"]))] +
                           [circuit.total_laps - (qr["pit_laps"][-1] if qr["pit_laps"] else 0)])),
            name=qr["strategy_name"],
        )
        # Reconstruct strategy from candidate list
        matching = [c for c in candidates if c.name == qr["strategy_name"]]
        if matching:
            s = matching[0]
        
        r = evaluate_strategy(
            circuit, drivers, target_code, grid_position,
            s, ai_strategy, n_sims=n_sims,
        )
        full_results.append(r)
    
    # Best strategy
    full_results.sort(key=lambda r: r["median_pos"])
    best = full_results[0]
    
    # Extract compact result
    sample_races = best.get("sample_races", {})
    
    # Use median race for the position trace summary
    median_race = sample_races.get("median", {})
    position_trace = median_race.get("positions", [])
    
    # Subsample position trace for compact TS (every 3 laps + last lap)
    if position_trace:
        sampled = position_trace[::3]
        if position_trace[-1] != sampled[-1]:
            sampled.append(position_trace[-1])
    else:
        sampled = []
    
    return {
        "strategy_name": best["strategy_name"],
        "compounds": best["compound_sequence"],
        "pit_laps": best["pit_laps"],
        "stops": best["num_stops"],
        "median_pos": round(best["median_pos"], 1),
        "mean_pos": round(best["mean_pos"], 1),
        "p5_pos": round(best["p5_pos"], 1),
        "p95_pos": round(best["p95_pos"], 1),
        "position_trace": sampled,
        "sample_races": sample_races,
    }


def precompute_circuit(
    circuit_key: str,
    circuit_params: CircuitParams,
    drivers: list[DriverConfig],
    n_sims: int = 50,
) -> dict:
    """Pre-compute all driver × grid position combos for one circuit."""
    logger.info(f"{'=' * 60}")
    logger.info(f"  PRECOMPUTING: {circuit_params.circuit_name}")
    logger.info(f"  {len(drivers)} drivers × 20 positions = {len(drivers) * 20} combos")
    logger.info(f"{'=' * 60}")
    
    t0 = time.time()
    results = {}
    total = len(drivers) * 20
    done = 0
    
    for driver in drivers:
        driver_results = {}
        for grid_pos in range(1, 21):
            try:
                r = optimize_scenario(
                    circuit_params, drivers, driver.code, 
                    grid_pos, n_sims=n_sims,
                )
                driver_results[str(grid_pos)] = r
            except Exception as e:
                logger.error(f"  FAILED: {driver.code} P{grid_pos}: {e}")
                driver_results[str(grid_pos)] = {
                    "strategy_name": "MH default",
                    "compounds": ["MEDIUM", "HARD"],
                    "pit_laps": [int(circuit_params.total_laps * 0.5)],
                    "stops": 1,
                    "median_pos": float(grid_pos),
                    "mean_pos": float(grid_pos),
                    "p5_pos": max(1.0, grid_pos - 3.0),
                    "p95_pos": min(20.0, grid_pos + 3.0),
                    "position_trace": [],
                    "sample_race": None,
                }
            
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate
                logger.info(
                    f"  {done}/{total} ({100*done/total:.0f}%) "
                    f"— {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
                )
        
        results[driver.code] = driver_results
    
    elapsed = time.time() - t0
    logger.info(f"  ✓ {circuit_params.circuit_name}: {total} combos in {elapsed:.0f}s")
    
    return {
        "circuit_key": circuit_key,
        "circuit_name": circuit_params.circuit_name,
        "total_laps": circuit_params.total_laps,
        "drivers": results,
    }


def write_detail_json(all_results: list, out_dir: str = "frontend/public/scenarios"):
    """
    Write per-circuit JSON files with full sample_races data.
    These are fetched on demand by the frontend.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for circuit_data in all_results:
        key = circuit_data["circuit_key"]
        detail = {}
        
        for driver_code, positions in circuit_data["drivers"].items():
            detail[driver_code] = {}
            for grid_pos, result in positions.items():
                sr = result.get("sample_races", {})
                detail[driver_code][grid_pos] = sr
        
        path = out_path / f"{key}.json"
        with open(path, "w") as f:
            json.dump(detail, f)
        
        size_kb = path.stat().st_size / 1024
        logger.info(f"  Detail JSON: {path} ({size_kb:.0f} KB)")


def write_typescript(all_results: list, out_dir: str = "frontend/src/data"):
    """
    Write compact TS file with stats only (no sample_races).
    Detailed race data is in per-circuit JSON files loaded on demand.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("// Auto-generated by precompute_scenarios.py")
    lines.append("// Do not edit manually")
    lines.append("")
    
    # Types
    lines.append("export interface SampleRace {")
    lines.append("  positions: number[];")
    lines.append("  tyre_ages: number[];")
    lines.append("  compounds: string[];")
    lines.append("  pit_laps: number[];")
    lines.append("  sc_laps: number[];")
    lines.append("  target_position: number;")
    lines.append("  narrative: string;")
    lines.append("}")
    lines.append("")
    lines.append("export interface SampleRaces {")
    lines.append("  best: SampleRace;")
    lines.append("  worst: SampleRace;")
    lines.append("  median: SampleRace;")
    lines.append("  early_sc?: SampleRace;")
    lines.append("  late_sc?: SampleRace;")
    lines.append("}")
    lines.append("")
    lines.append("export interface ScenarioResult {")
    lines.append("  strategy: string;")
    lines.append("  compounds: string[];")
    lines.append("  pitLaps: number[];")
    lines.append("  stops: number;")
    lines.append("  medianPos: number;")
    lines.append("  meanPos: number;")
    lines.append("  bestPos: number;")
    lines.append("  worstPos: number;")
    lines.append("  positionTrace: number[];")
    lines.append("}")
    lines.append("")
    lines.append("export interface CircuitScenarios {")
    lines.append("  circuitKey: string;")
    lines.append("  circuitName: string;")
    lines.append("  totalLaps: number;")
    lines.append("  drivers: Record<string, Record<string, ScenarioResult>>;")
    lines.append("}")
    lines.append("")
    lines.append("export const scenarioData: Record<string, CircuitScenarios> = {")
    
    for circuit_data in all_results:
        key = circuit_data["circuit_key"]
        name = circuit_data["circuit_name"].replace('"', '\\"')
        
        lines.append(f'  "{key}": {{')
        lines.append(f'    circuitKey: "{key}",')
        lines.append(f'    circuitName: "{name}",')
        lines.append(f'    totalLaps: {circuit_data["total_laps"]},')
        lines.append(f'    drivers: {{')
        
        for driver_code, positions in circuit_data["drivers"].items():
            lines.append(f'      "{driver_code}": {{')
            for grid_pos, result in positions.items():
                compounds = json.dumps(result["compounds"])
                pit_laps = json.dumps(result["pit_laps"])
                trace = json.dumps(result.get("position_trace", []))
                
                lines.append(
                    f'        "{grid_pos}": {{ strategy: "{result["strategy_name"]}", '
                    f'compounds: {compounds}, pitLaps: {pit_laps}, '
                    f'stops: {result["stops"]}, '
                    f'medianPos: {result["median_pos"]}, '
                    f'meanPos: {result["mean_pos"]}, '
                    f'bestPos: {result["p5_pos"]}, '
                    f'worstPos: {result["p95_pos"]}, '
                    f'positionTrace: {trace} }},'
                )
            lines.append(f'      }},')
        
        lines.append(f'    }},')
        lines.append(f'  }},')
    
    lines.append("}")
    lines.append("")
    
    ts_path = out_path / "scenarios.ts"
    with open(ts_path, "w") as f:
        f.write("\n".join(lines))
    
    size_kb = ts_path.stat().st_size / 1024
    logger.info(f"✓ Written: {ts_path} ({len(all_results)} circuits, {size_kb:.0f} KB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-compute race strategy scenarios")
    parser.add_argument("--circuits", nargs="+", default=["all"])
    parser.add_argument("--n-sims", type=int, default=50)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--output-dir", default="results/scenarios")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    # Load data
    drivers, teams, overtaking = load_drivers()
    config = load_config(args.config)
    
    # Optionally load XGBoost deg model
    deg_model = None
    feature_cols = None
    try:
        import xgboost as xgb
        deg_model = xgb.XGBRegressor()
        deg_model.load_model("models/tyre_deg_production.json")
        with open("models/comparison_results.json") as f:
            comp = json.load(f)
        feature_cols = comp["experiment"]["feature_columns"]
        logger.info("Loaded XGBoost degradation model")
    except Exception as e:
        logger.warning(f"Could not load deg model, using defaults: {e}")
    
    # Determine circuits
    available = get_available_circuits(args.season)
    if "all" in args.circuits:
        circuit_keys = available
    else:
        circuit_keys = [k for k in args.circuits if k in available]
        missing = [k for k in args.circuits if k not in available]
        if missing:
            logger.warning(f"No MC results for: {missing}")
    
    logger.info(f"Circuits: {circuit_keys}")
    logger.info(f"Drivers: {len(drivers)}")
    logger.info(f"Sims per combo: {args.n_sims}")
    logger.info(f"Total combos: {len(circuit_keys) * len(drivers) * 20}")
    
    # Pre-compute
    all_results = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for circuit_key in circuit_keys:
        circuit = load_circuit_as_params(
            circuit_key, args.season, config, overtaking,
            deg_model=deg_model, feature_cols=feature_cols,
        )
        
        result = precompute_circuit(circuit_key, circuit, drivers, n_sims=args.n_sims)
        all_results.append(result)
        
        # Save per-circuit JSON (with sample_race for full chart data)
        out_path = out_dir / f"scenarios_{circuit_key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        logger.info(f"  Saved: {out_path}")
    
    # Write TypeScript (compact stats)
    write_typescript(all_results)
    
    # Write per-circuit JSON (full sample_races for frontend)
    write_detail_json(all_results)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  COMPLETE: {len(all_results)} circuits pre-computed")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()