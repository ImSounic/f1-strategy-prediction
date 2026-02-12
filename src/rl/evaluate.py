"""
RL vs Monte Carlo Comparison
===============================
Evaluate trained RL agents against the MC-optimized strategies
on the same stochastic race seeds for fair comparison.

For each circuit:
    1. Load trained PPO agent
    2. Load precomputed MC best strategy
    3. Run both on N identical random seeds
    4. Compare total race times, stop counts, SC response quality
    5. Export results for frontend visualization

Usage:
    python -m src.rl.evaluate --circuit bahrain --n-races 500
    python -m src.rl.evaluate --all --n-races 300
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from stable_baselines3 import PPO

from src.rl.environment import F1StrategyEnv
from src.simulation.strategy_simulator import (
    load_circuit_config,
    generate_strategies,
    simulate_race_fast,
    precompute_deg_rates,
    predict_degradation,
    COMPOUND_HARDNESS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_strategy_name(raw: str) -> str:
    name = re.sub(r'\s*\(\d+(?:/\d+)*\)\s*', '', raw)
    name = name.replace('MEDIUM', 'M').replace('HARD', 'H').replace('SOFT', 'S')
    return name.strip()


def run_mc_strategy(strategy, circuit, deg_model, feature_cols, fuel_config, seed):
    """Run a single MC race for a fixed strategy with a specific seed."""
    from src.simulation.strategy_simulator import precompute_deg_rates
    
    rng = np.random.default_rng(seed)
    
    start_fuel = fuel_config["start_fuel_kg"]
    fuel_effect = fuel_config["fuel_effect_per_kg_seconds"]
    burn_rate = start_fuel / circuit.total_laps
    base_pace = 90.0
    
    # SC probabilities per lap
    if circuit.sc_prob_per_race > 0:
        sc_prob_per_lap = 1 - (1 - circuit.sc_prob_per_race) ** (1 / circuit.total_laps)
    else:
        sc_prob_per_lap = 0.0
    if circuit.vsc_prob_per_race > 0:
        vsc_prob_per_lap = 1 - (1 - circuit.vsc_prob_per_race) ** (1 / circuit.total_laps)
    else:
        vsc_prob_per_lap = 0.0

    # Precompute deg rates
    deg_rates = precompute_deg_rates(strategy, circuit, deg_model, feature_cols)

    total_time, sc_count = simulate_race_fast(
        strategy, circuit, deg_rates, fuel_config,
        sc_prob_per_lap, vsc_prob_per_lap, base_pace,
        burn_rate, start_fuel, fuel_effect, rng,
    )
    return total_time, sc_count


def run_rl_agent(model, env, seed):
    """Run one race with the RL agent using a specific seed."""
    obs, info = env.reset(seed=seed)
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    return info


def compare_circuit(
    circuit_key: str,
    season: int = 2025,
    n_races: int = 500,
    config_path: str = "configs/config.yaml",
) -> dict:
    """
    Head-to-head comparison: RL agent vs MC best strategy.
    Both run on identical random seeds for fair comparison.
    """
    config = load_config(config_path)
    raw_paths = config["paths"]["raw"]
    fuel_config = config["modeling"]["fuel_model"]
    
    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"
    
    logger.info("=" * 65)
    logger.info(f"  RL vs MC COMPARISON — {circuit_key.upper()}")
    logger.info("=" * 65)
    
    # Load circuit
    circuit = load_circuit_config(circuit_key, season, circuit_csv, sc_priors_path, weather_dir)
    
    # Load deg model
    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]
    
    # Load MC best strategy from precomputed results
    results_path = Path("results") / f"strategy_{circuit_key}_{season}.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No precomputed MC results: {results_path}")
    
    with open(results_path) as f:
        mc_data = json.load(f)
    
    mc_best = mc_data["rankings"][0]
    mc_best_name = mc_best["strategy_name"]
    
    # Reconstruct the strategy object for simulation
    strategies = generate_strategies(circuit)
    mc_strategy = None
    for s in strategies:
        if s.name == mc_best_name:
            mc_strategy = s
            break
    
    if mc_strategy is None:
        # Fallback: find closest match
        for s in strategies:
            clean = clean_strategy_name(s.name)
            if clean == clean_strategy_name(mc_best_name):
                mc_strategy = s
                break
    
    if mc_strategy is None:
        raise ValueError(f"Could not find MC strategy: {mc_best_name}")
    
    logger.info(f"  MC best:  {clean_strategy_name(mc_best_name)}")
    
    # Load RL agent
    model_path = Path("models/rl") / f"ppo_{circuit_key}_{season}"
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"No trained RL agent: {model_path}")
    
    rl_model = PPO.load(str(model_path))
    
    # Build RL environment
    env = F1StrategyEnv(circuit, deg_model, feature_cols, fuel_config)
    
    logger.info(f"  Races:    {n_races}")
    logger.info(f"\n  Running head-to-head comparison...")
    
    # ── Run both on same seeds ──
    t0 = time.time()
    
    mc_times = []
    mc_sc_counts = []
    rl_times = []
    rl_stops = []
    rl_compounds = []
    rl_sc_counts = []
    rl_pit_laps_all = []
    rl_wins = 0
    mc_wins = 0
    
    # Per-scenario tracking
    sc_race_rl_wins = 0
    sc_race_total = 0
    no_sc_race_rl_wins = 0
    no_sc_race_total = 0
    
    for i in range(n_races):
        seed = 5000 + i  # deterministic seed sequence
        
        # MC run
        mc_time, mc_sc = run_mc_strategy(
            mc_strategy, circuit, deg_model, feature_cols, fuel_config, seed
        )
        mc_times.append(mc_time)
        mc_sc_counts.append(mc_sc)
        
        # RL run (same seed)
        rl_info = run_rl_agent(rl_model, env, seed)
        rl_time = rl_info["total_time"]
        rl_times.append(rl_time)
        rl_stops.append(rl_info["stops_done"])
        rl_compounds.append(len(rl_info["compounds_used"]))
        rl_sc_counts.append(rl_info["sc_events"])
        rl_pit_laps_all.append(rl_info["history"]["pit_laps"])
        
        # Win tracking
        if rl_time < mc_time:
            rl_wins += 1
        else:
            mc_wins += 1
        
        # SC vs no-SC breakdown
        had_sc = mc_sc > 0 or rl_info["sc_events"] > 0
        if had_sc:
            sc_race_total += 1
            if rl_time < mc_time:
                sc_race_rl_wins += 1
        else:
            no_sc_race_total += 1
            if rl_time < mc_time:
                no_sc_race_rl_wins += 1
    
    elapsed = time.time() - t0
    
    # ── Compute stats ──
    mc_times = np.array(mc_times)
    rl_times = np.array(rl_times)
    
    result = {
        "circuit_key": circuit_key,
        "circuit_name": circuit.circuit_name,
        "season": season,
        "n_races": n_races,
        "elapsed_seconds": round(elapsed, 1),
        "mc_strategy": clean_strategy_name(mc_best_name),
        "mc_strategy_stops": mc_strategy.num_stops,
        "mc_compound_sequence": mc_strategy.compound_sequence,
        "mc": {
            "median_time": round(float(np.median(mc_times)), 1),
            "mean_time": round(float(np.mean(mc_times)), 1),
            "std_time": round(float(np.std(mc_times)), 1),
            "p5_time": round(float(np.percentile(mc_times, 5)), 1),
            "p95_time": round(float(np.percentile(mc_times, 95)), 1),
            "mean_sc": round(float(np.mean(mc_sc_counts)), 2),
        },
        "rl": {
            "median_time": round(float(np.median(rl_times)), 1),
            "mean_time": round(float(np.mean(rl_times)), 1),
            "std_time": round(float(np.std(rl_times)), 1),
            "p5_time": round(float(np.percentile(rl_times, 5)), 1),
            "p95_time": round(float(np.percentile(rl_times, 95)), 1),
            "mean_stops": round(float(np.mean(rl_stops)), 2),
            "stop_distribution": {
                str(s): int(np.sum(np.array(rl_stops) == s))
                for s in sorted(set(rl_stops))
            },
            "mean_compounds": round(float(np.mean(rl_compounds)), 2),
            "mean_sc": round(float(np.mean(rl_sc_counts)), 2),
            "legal_pct": round(100 * float(np.mean(np.array(rl_compounds) >= 2)), 1),
        },
        "comparison": {
            "rl_win_rate": round(100 * rl_wins / n_races, 1),
            "mc_win_rate": round(100 * mc_wins / n_races, 1),
            "median_delta": round(float(np.median(rl_times) - np.median(mc_times)), 2),
            "mean_delta": round(float(np.mean(rl_times) - np.mean(mc_times)), 2),
            "rl_advantage_seconds": round(float(np.median(mc_times) - np.median(rl_times)), 2),
            # SC breakdown
            "sc_races": sc_race_total,
            "sc_race_rl_win_rate": round(100 * sc_race_rl_wins / max(sc_race_total, 1), 1),
            "no_sc_races": no_sc_race_total,
            "no_sc_race_rl_win_rate": round(100 * no_sc_race_rl_wins / max(no_sc_race_total, 1), 1),
        },
        # Sample RL lap-by-lap decisions (categorised for frontend)
        "sample_races": [],
    }
    
    # Collect diverse sample races by category
    # Categorise the 500 races we already ran
    race_categories = []
    for i in range(n_races):
        sc_count = int(mc_sc_counts[i]) + int(rl_sc_counts[i] if i < len(rl_sc_counts) else 0)
        # Use the RL SC count from rl_info since both share the same seed
        rl_sc = int(rl_sc_counts[i]) if i < len(rl_sc_counts) else 0
        rl_won = bool(rl_times[i] < mc_times[i])
        race_categories.append({
            "index": i,
            "seed": 5000 + i,
            "sc_count": rl_sc,
            "rl_won": rl_won,
            "delta": float(mc_times[i] - rl_times[i]),
        })
    
    # Pick diverse representative races
    clean_races = [r for r in race_categories if r["sc_count"] == 0]
    sc1_races = [r for r in race_categories if r["sc_count"] == 1]
    sc2_races = [r for r in race_categories if r["sc_count"] >= 2]
    
    selected_seeds = []
    
    # 1) Clean Race
    if clean_races:
        # Pick one near the median delta
        clean_races.sort(key=lambda r: abs(r["delta"]))
        selected_seeds.append(("Clean Race", clean_races[0]["seed"]))
    
    # 2) Clean Race — RL Win (if different from above)
    clean_rl_wins = [r for r in clean_races if r["rl_won"]]
    if clean_rl_wins and (not selected_seeds or clean_rl_wins[0]["seed"] != selected_seeds[0][1]):
        # Pick one with biggest RL advantage
        clean_rl_wins.sort(key=lambda r: r["delta"], reverse=True)
        selected_seeds.append(("Clean — RL Win", clean_rl_wins[0]["seed"]))
    
    # 3) 1 Safety Car
    if sc1_races:
        sc1_races.sort(key=lambda r: abs(r["delta"]))
        selected_seeds.append(("1 Safety Car", sc1_races[0]["seed"]))
    
    # 4) SC — RL Adapts (RL wins with SC present)
    sc_rl_wins = [r for r in sc1_races + sc2_races if r["rl_won"]]
    if sc_rl_wins:
        sc_rl_wins.sort(key=lambda r: r["delta"], reverse=True)
        seed_already = {s for _, s in selected_seeds}
        best = next((r for r in sc_rl_wins if r["seed"] not in seed_already), None)
        if best:
            selected_seeds.append(("SC — RL Adapts", best["seed"]))
    
    # 5) Multi Safety Car
    if sc2_races:
        seed_already = {s for _, s in selected_seeds}
        remaining = [r for r in sc2_races if r["seed"] not in seed_already]
        if remaining:
            remaining.sort(key=lambda r: abs(r["delta"]))
            selected_seeds.append(("Multi SC", remaining[0]["seed"]))
    
    # Fallback: if we have < 3 races, fill with sequential seeds
    if len(selected_seeds) < 3:
        for i in range(min(5, n_races)):
            seed = 5000 + i
            if seed not in {s for _, s in selected_seeds}:
                sc = int(rl_sc_counts[i]) if i < len(rl_sc_counts) else 0
                cat = "Clean Race" if sc == 0 else f"{sc} SC"
                selected_seeds.append((cat, seed))
                if len(selected_seeds) >= 5:
                    break
    
    # Compute MC tyre ages & pit laps from fixed strategy (deterministic)
    mc_tyre_ages = []
    mc_pit_laps = []
    mc_compounds_seq = []
    cumulative_laps = 0
    for stint in mc_strategy.stints:
        for lap_in_stint in range(1, stint.target_laps + 1):
            mc_tyre_ages.append(lap_in_stint)
            mc_compounds_seq.append(stint.compound)
        cumulative_laps += stint.target_laps
        if cumulative_laps < circuit.total_laps:
            mc_pit_laps.append(cumulative_laps)
    # Pad/trim to exact total_laps
    mc_tyre_ages = mc_tyre_ages[:circuit.total_laps]
    mc_compounds_seq = mc_compounds_seq[:circuit.total_laps]
    
    # Re-run RL agent on selected seeds to get full lap-by-lap data
    for category, seed in selected_seeds:
        rl_info = run_rl_agent(rl_model, env, seed)
        mc_time_for_seed, mc_sc_for_seed = run_mc_strategy(
            mc_strategy, circuit, deg_model, feature_cols, fuel_config, seed
        )
        result["sample_races"].append({
            "seed": seed,
            "category": category,
            "total_time": round(rl_info["total_time"], 1),
            "mc_time": round(mc_time_for_seed, 1),
            "rl_won": bool(rl_info["total_time"] < mc_time_for_seed),
            "stops": rl_info["stops_done"],
            "compounds": rl_info["history"]["compounds"],
            "pit_laps": rl_info["history"]["pit_laps"],
            "sc_laps": rl_info["history"]["sc_laps"],
            "vsc_laps": rl_info["history"]["vsc_laps"],
            "tyre_ages": rl_info["history"]["tyre_ages"],
            # MC overlay data (fixed strategy, same for all races)
            "mc_tyre_ages": mc_tyre_ages,
            "mc_pit_laps": mc_pit_laps,
            "mc_compounds": mc_compounds_seq,
        })
    
    env.close()
    
    # ── Print summary ──
    logger.info(f"\n  {'─' * 55}")
    logger.info(f"  {'RESULTS':^55}")
    logger.info(f"  {'─' * 55}")
    logger.info(f"  {'':20} {'MC':>15} {'RL':>15}")
    logger.info(f"  {'Median time':20} {np.median(mc_times):>14.1f}s {np.median(rl_times):>14.1f}s")
    logger.info(f"  {'Std time':20} {np.std(mc_times):>14.1f}s {np.std(rl_times):>14.1f}s")
    logger.info(f"  {'Mean stops':20} {mc_strategy.num_stops:>15d} {np.mean(rl_stops):>14.2f}")
    logger.info(f"  {'Mean SC events':20} {np.mean(mc_sc_counts):>14.2f} {np.mean(rl_sc_counts):>14.2f}")
    logger.info(f"  {'─' * 55}")
    logger.info(f"  RL advantage:  {result['comparison']['rl_advantage_seconds']:+.2f}s (median)")
    logger.info(f"  RL win rate:   {result['comparison']['rl_win_rate']:.1f}%")
    logger.info(f"    SC races:    {result['comparison']['sc_race_rl_win_rate']:.1f}% "
                f"({sc_race_total} races)")
    logger.info(f"    No-SC races: {result['comparison']['no_sc_race_rl_win_rate']:.1f}% "
                f"({no_sc_race_total} races)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare RL agent vs MC strategy")
    parser.add_argument("--circuit", type=str, help="Evaluate specific circuit")
    parser.add_argument("--all", action="store_true", help="Evaluate all trained circuits")
    parser.add_argument("--n-races", type=int, default=500)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if args.all:
        # Find all trained models
        model_dir = Path("models/rl")
        model_files = list(model_dir.glob(f"ppo_*_{args.season}.zip"))
        circuit_keys = [f.stem.replace(f"ppo_", "").replace(f"_{args.season}", "")
                       for f in model_files]
        
        logger.info(f"Found {len(circuit_keys)} trained agents")
        
        all_results = []
        for key in sorted(circuit_keys):
            try:
                result = compare_circuit(key, args.season, args.n_races, args.config)
                
                # Save individual result
                out_path = results_dir / f"rl_comparison_{key}_{args.season}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"  ✓ Saved: {out_path}")
                
                all_results.append(result)
            except Exception as e:
                logger.error(f"  ✗ {key}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save combined summary
        summary = {
            "n_circuits": len(all_results),
            "n_races_per_circuit": args.n_races,
            "season": args.season,
            "circuits": all_results,
            "aggregate": {
                "mean_rl_advantage": round(
                    float(np.mean([r["comparison"]["rl_advantage_seconds"] for r in all_results])), 2
                ),
                "mean_rl_win_rate": round(
                    float(np.mean([r["comparison"]["rl_win_rate"] for r in all_results])), 1
                ),
            }
        }
        
        summary_path = results_dir / f"rl_comparison_summary_{args.season}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\n✓ Summary: {summary_path}")
        
        # Export to TypeScript
        export_rl_typescript(all_results, args.season)
    
    elif args.circuit:
        result = compare_circuit(args.circuit, args.season, args.n_races, args.config)
        
        out_path = results_dir / f"rl_comparison_{args.circuit}_{args.season}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"\n  ✓ Saved: {out_path}")
        
        export_rl_typescript([result], args.season)
    
    else:
        parser.print_help()


def export_rl_typescript(results: list, season: int):
    """Export RL comparison data to TypeScript for the frontend."""
    lines = []
    
    # Types
    lines.append("// Auto-generated by src/rl/evaluate.py")
    lines.append("// Do not edit manually")
    lines.append("")
    lines.append("export interface RLSampleRace {")
    lines.append("  seed: number;")
    lines.append("  category: string;")
    lines.append("  totalTime: number;")
    lines.append("  mcTime: number;")
    lines.append("  rlWon: boolean;")
    lines.append("  stops: number;")
    lines.append("  compounds: string[];")
    lines.append("  pitLaps: number[];")
    lines.append("  scLaps: number[];")
    lines.append("  vscLaps: number[];")
    lines.append("  tyreAges: number[];")
    lines.append("  mcTyreAges: number[];")
    lines.append("  mcPitLaps: number[];")
    lines.append("  mcCompounds: string[];")
    lines.append("}")
    lines.append("")
    lines.append("export interface RLStats {")
    lines.append("  medianTime: number;")
    lines.append("  meanTime: number;")
    lines.append("  stdTime: number;")
    lines.append("  p5Time: number;")
    lines.append("  p95Time: number;")
    lines.append("  meanStops?: number;")
    lines.append("  meanSc: number;")
    lines.append("  legalPct?: number;")
    lines.append("  stopDistribution?: Record<string, number>;")
    lines.append("}")
    lines.append("")
    lines.append("export interface RLComparison {")
    lines.append("  rlWinRate: number;")
    lines.append("  mcWinRate: number;")
    lines.append("  medianDelta: number;")
    lines.append("  rlAdvantageSeconds: number;")
    lines.append("  scRaces: number;")
    lines.append("  scRaceRlWinRate: number;")
    lines.append("  noScRaces: number;")
    lines.append("  noScRaceRlWinRate: number;")
    lines.append("}")
    lines.append("")
    lines.append("export interface CircuitRLResult {")
    lines.append("  circuitKey: string;")
    lines.append("  circuitName: string;")
    lines.append("  season: number;")
    lines.append("  nRaces: number;")
    lines.append("  mcStrategy: string;")
    lines.append("  mcStops: number;")
    lines.append("  mc: RLStats;")
    lines.append("  rl: RLStats;")
    lines.append("  comparison: RLComparison;")
    lines.append("  sampleRaces: RLSampleRace[];")
    lines.append("}")
    lines.append("")
    lines.append("export const rlResults: Record<string, CircuitRLResult> = {")
    
    for r in sorted(results, key=lambda x: x["circuit_key"]):
        key = r["circuit_key"]
        cname = r["circuit_name"].replace('"', '\\"')
        mc = r["mc"]
        rl = r["rl"]
        comp = r["comparison"]
        
        lines.append(f'  "{key}": {{')
        lines.append(f'    circuitKey: "{key}",')
        lines.append(f'    circuitName: "{cname}",')
        lines.append(f'    season: {r["season"]},')
        lines.append(f'    nRaces: {r["n_races"]},')
        lines.append(f'    mcStrategy: "{r["mc_strategy"]}",')
        lines.append(f'    mcStops: {r["mc_strategy_stops"]},')
        
        # MC stats
        lines.append(f'    mc: {{ medianTime: {mc["median_time"]}, meanTime: {mc["mean_time"]}, '
                     f'stdTime: {mc["std_time"]}, p5Time: {mc["p5_time"]}, '
                     f'p95Time: {mc["p95_time"]}, meanSc: {mc["mean_sc"]} }},')
        
        # RL stats
        stop_dist = json.dumps(rl.get("stop_distribution", {}))
        lines.append(f'    rl: {{ medianTime: {rl["median_time"]}, meanTime: {rl["mean_time"]}, '
                     f'stdTime: {rl["std_time"]}, p5Time: {rl["p5_time"]}, '
                     f'p95Time: {rl["p95_time"]}, meanStops: {rl["mean_stops"]}, '
                     f'meanSc: {rl["mean_sc"]}, legalPct: {rl["legal_pct"]}, '
                     f"stopDistribution: {stop_dist} }},")
        
        # Comparison
        lines.append(f'    comparison: {{ rlWinRate: {comp["rl_win_rate"]}, '
                     f'mcWinRate: {comp["mc_win_rate"]}, '
                     f'medianDelta: {comp["median_delta"]}, '
                     f'rlAdvantageSeconds: {comp["rl_advantage_seconds"]}, '
                     f'scRaces: {comp["sc_races"]}, '
                     f'scRaceRlWinRate: {comp["sc_race_rl_win_rate"]}, '
                     f'noScRaces: {comp["no_sc_races"]}, '
                     f'noScRaceRlWinRate: {comp["no_sc_race_rl_win_rate"]} }},')
        
        # Sample races
        lines.append(f'    sampleRaces: [')
        for sr in r.get("sample_races", []):
            compounds_str = json.dumps(sr["compounds"])
            pit_str = json.dumps(sr["pit_laps"])
            sc_str = json.dumps(sr["sc_laps"])
            vsc_str = json.dumps(sr["vsc_laps"])
            ages_str = json.dumps(sr["tyre_ages"])
            mc_ages_str = json.dumps(sr.get("mc_tyre_ages", []))
            mc_pit_str = json.dumps(sr.get("mc_pit_laps", []))
            mc_comp_str = json.dumps(sr.get("mc_compounds", []))
            category = sr.get("category", "Race")
            mc_time = sr.get("mc_time", 0)
            rl_won = "true" if sr.get("rl_won", False) else "false"
            lines.append(f'      {{ seed: {sr["seed"]}, category: "{category}", '
                        f'totalTime: {sr["total_time"]}, mcTime: {mc_time}, rlWon: {rl_won}, '
                        f'stops: {sr["stops"]}, compounds: {compounds_str}, '
                        f'pitLaps: {pit_str}, scLaps: {sc_str}, '
                        f'vscLaps: {vsc_str}, tyreAges: {ages_str}, '
                        f'mcTyreAges: {mc_ages_str}, mcPitLaps: {mc_pit_str}, '
                        f'mcCompounds: {mc_comp_str} }},')
        lines.append(f'    ],')
        
        lines.append(f'  }},')
    
    lines.append("}")
    lines.append("")
    
    # Write
    out_path = Path("frontend/src/data/rl.ts")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"✓ Written: {out_path} ({len(results)} circuits)")


if __name__ == "__main__":
    main()