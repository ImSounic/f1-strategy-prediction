"""
Scenario-Based Contingency Planner
====================================
Extends the Monte Carlo simulator with conditional scenarios.

Instead of averaging over all possible SC timings and degradation outcomes,
this engine forces specific conditions and asks: "Given that X happens,
what is the best strategy?"

Scenarios are defined as race condition overrides:
    - forced_sc_window: force a safety car within a lap range
    - suppress_sc: prevent any SC from occurring
    - deg_multiplier: scale degradation (>1 = worse than predicted)
    - forced_vsc_window: force a virtual safety car

For each scenario × strategy combination, runs N simulations with the
forced conditions (plus remaining stochastic elements) to produce
a strategy ranking specific to that scenario.

Output:
    Per-circuit: list of scenarios, each with strategy rankings,
    decision triggers, and time savings vs the unconditioned plan.

Usage:
    python -m src.simulation.scenario_engine --circuit bahrain --season 2025
    python -m src.simulation.scenario_engine --all --season 2025
"""

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
import yaml

from src.simulation.strategy_simulator import (
    CircuitConfig,
    Strategy,
    StintPlan,
    load_circuit_config,
    generate_strategies,
    precompute_deg_rates,
    predict_degradation,
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


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """A specific race condition to simulate."""
    id: str
    name: str
    description: str
    icon: str
    # Condition overrides
    forced_sc_window: Optional[tuple] = None   # (start_lap, end_lap) — force SC here
    forced_vsc_window: Optional[tuple] = None  # (start_lap, end_lap) — force VSC here
    suppress_sc: bool = False                  # prevent any SC
    deg_multiplier: float = 1.0                # scale degradation (1.0 = normal)
    # Metadata
    base_probability: float = 0.0              # estimated frequency from data


def generate_scenarios(circuit: CircuitConfig) -> list:
    """
    Generate the most relevant race scenarios for a given circuit.
    
    Probabilities are derived from:
        - Circuit SC history (Bayesian priors)
        - General F1 statistics (2022-2025)
    """
    total = circuit.total_laps
    sc_prob = circuit.sc_prob_per_race
    vsc_prob = getattr(circuit, 'vsc_prob_per_race', sc_prob * 0.5)
    
    # Define lap windows relative to race distance
    early_end = max(5, int(total * 0.25))
    mid_start = early_end + 1
    mid_end = int(total * 0.60)
    late_start = mid_end + 1
    late_end = total - 3  # SC can't happen in last 2 laps
    
    # Base probabilities from F1 data (2022-2025 averages):
    #   ~55% of races have at least 1 SC
    #   SC distribution: ~30% early, ~40% mid, ~30% late
    #   ~25% have VSC only (no full SC)
    #   ~12% have 2+ SC events
    
    p_any_sc = sc_prob
    p_no_sc = 1.0 - p_any_sc
    
    scenarios = [
        # ── No incidents ──
        Scenario(
            id="clean_race",
            name="Clean Race — No Safety Car",
            description=(
                f"No safety car or VSC throughout the race. "
                f"Pure pace strategy. Historically happens {p_no_sc:.0%} at this circuit."
            ),
            icon="🟢",
            suppress_sc=True,
            base_probability=round(p_no_sc, 2),
        ),
        
        # ── Early SC ──
        Scenario(
            id="early_sc",
            name=f"Early Safety Car (Lap 1–{early_end})",
            description=(
                f"SC deployed in the opening phase. Common from lap 1 incidents. "
                f"Creates an early free pit stop window — changes the entire race."
            ),
            icon="🔴",
            forced_sc_window=(1, early_end),
            base_probability=round(p_any_sc * 0.30, 2),
        ),
        
        # ── Mid-race SC ──
        Scenario(
            id="mid_sc",
            name=f"Mid-Race Safety Car (Lap {mid_start}–{mid_end})",
            description=(
                f"SC during the primary pit window. Splits the field between "
                f"those who have pitted and those who haven't."
            ),
            icon="🟡",
            forced_sc_window=(mid_start, mid_end),
            base_probability=round(p_any_sc * 0.40, 2),
        ),
        
        # ── Late SC ──
        Scenario(
            id="late_sc",
            name=f"Late Safety Car (Lap {late_start}–{late_end})",
            description=(
                f"SC in the final phase. Can force an unplanned extra stop "
                f"or give leaders a free pit. Dramatic strategy implications."
            ),
            icon="🟠",
            forced_sc_window=(late_start, late_end),
            base_probability=round(p_any_sc * 0.30, 2),
        ),
        
        # ── VSC only ──
        Scenario(
            id="vsc_only",
            name="Virtual Safety Car Only",
            description=(
                "VSC deployed but no full SC. Smaller time gain from pitting "
                "under VSC (~40% of SC saving). Rewards opportunistic stops."
            ),
            icon="🔵",
            forced_vsc_window=(mid_start, mid_end),
            suppress_sc=True,
            base_probability=round(min(vsc_prob, 0.25), 2),
        ),
        
        # ── Double SC ──
        Scenario(
            id="double_sc",
            name="Double Safety Car",
            description=(
                "Two separate SC events. Chaos scenario — 2-stop strategies "
                "get two free pit windows. Highly favourable for aggressive approaches."
            ),
            icon="🔴🔴",
            forced_sc_window=(1, early_end),  # first SC early
            # We'll inject a second SC in the simulation loop
            base_probability=round(p_any_sc * 0.12, 2),
        ),
        
        # ── High degradation ──
        Scenario(
            id="high_deg",
            name="Higher Than Expected Degradation",
            description=(
                "Tyre deg is 30% worse than the model predicts. Track temp higher, "
                "graining, or unexpected abrasion. Favours multi-stop strategies."
            ),
            icon="🌡️",
            deg_multiplier=1.30,
            base_probability=0.20,
        ),
        
        # ── Low degradation ──
        Scenario(
            id="low_deg",
            name="Lower Than Expected Degradation",
            description=(
                "Tyre deg is 25% better than predicted. Track evolution, "
                "cooler conditions, or rubber build-up. Favours extending stints."
            ),
            icon="❄️",
            deg_multiplier=0.75,
            base_probability=0.20,
        ),
        
        # ── Early SC + high deg combo ──
        Scenario(
            id="early_sc_high_deg",
            name="Early SC + High Degradation",
            description=(
                "Worst-case combo: incident in opening laps AND tyres wearing "
                "faster than expected. Strongly favours 2-stop strategies."
            ),
            icon="⚠️",
            forced_sc_window=(1, early_end),
            deg_multiplier=1.30,
            base_probability=round(p_any_sc * 0.30 * 0.20, 2),
        ),
        
        # ── Clean race + low deg (best case for 1-stop) ──
        Scenario(
            id="clean_low_deg",
            name="Clean Race + Low Degradation",
            description=(
                "Best-case for conservative strategies: no interruptions and "
                "tyres lasting longer than expected. 1-stop heaven."
            ),
            icon="✨",
            suppress_sc=True,
            deg_multiplier=0.75,
            base_probability=round(p_no_sc * 0.20, 2),
        ),
    ]
    
    # Normalise probabilities to sum to ~1.0
    total_prob = sum(s.base_probability for s in scenarios)
    if total_prob > 0:
        for s in scenarios:
            s.base_probability = round(s.base_probability / total_prob, 3)
    
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
#  CONDITIONED SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_race_conditioned(
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
    # ── Scenario overrides ──
    forced_sc_window: tuple = None,
    forced_vsc_window: tuple = None,
    suppress_sc: bool = False,
    deg_multiplier: float = 1.0,
    second_sc_window: tuple = None,
) -> tuple:
    """
    Race simulation with forced conditions.
    
    Based on simulate_race_fast but with scenario overrides:
        - forced_sc_window: inject SC at random lap within [start, end]
        - forced_vsc_window: inject VSC at random lap within [start, end]
        - suppress_sc: prevent any stochastic SC/VSC events
        - deg_multiplier: scale all degradation rates
        - second_sc_window: inject a second SC event (for double SC scenario)
    """
    total_laps = circuit.total_laps
    total_time = 0.0
    sc_count = 0
    sc_remaining = 0
    vsc_remaining = 0
    current_stint = 0
    laps_in_stint = 0
    
    # Pre-determine forced SC/VSC laps
    forced_sc_lap = None
    if forced_sc_window:
        lo, hi = forced_sc_window
        lo = max(1, min(lo, total_laps - 4))
        hi = max(lo, min(hi, total_laps - 4))
        forced_sc_lap = int(rng.integers(lo, hi + 1))
    
    forced_vsc_lap = None
    if forced_vsc_window:
        lo, hi = forced_vsc_window
        lo = max(1, min(lo, total_laps - 3))
        hi = max(lo, min(hi, total_laps - 3))
        forced_vsc_lap = int(rng.integers(lo, hi + 1))
    
    second_sc_lap = None
    if second_sc_window:
        lo, hi = second_sc_window
        lo = max(1, min(lo, total_laps - 4))
        hi = max(lo, min(hi, total_laps - 4))
        second_sc_lap = int(rng.integers(lo, hi + 1))
    
    for lap in range(1, total_laps + 1):
        stint = strategy.stints[current_stint]
        laps_in_stint += 1
        
        # Fuel
        fuel_remaining = max(0, start_fuel - burn_rate * (lap - 1))
        fuel_time = fuel_remaining * fuel_effect
        
        # Degradation (with scenario multiplier)
        deg_rate = precomputed_deg_rates[current_stint] + rng.normal(0, 0.01)
        deg_rate = max(0.005, deg_rate) * deg_multiplier
        tyre_deg = deg_rate * laps_in_stint + 0.002 * (laps_in_stint ** 1.3)
        
        # ── SC/VSC Logic ──
        if sc_remaining > 0:
            lap_time = base_pace * 1.40
            sc_remaining -= 1
        elif vsc_remaining > 0:
            lap_time = base_pace * 1.20
            vsc_remaining -= 1
        else:
            # Normal lap
            lap_time = base_pace + fuel_time + tyre_deg + rng.normal(0, 0.3)
            
            # Check for forced SC
            if forced_sc_lap and lap == forced_sc_lap and sc_remaining == 0:
                sc_remaining = int(rng.integers(3, 7))
                lap_time = base_pace * 1.40
                sc_count += 1
                forced_sc_lap = None  # consumed
            # Check for second forced SC
            elif second_sc_lap and lap == second_sc_lap and sc_remaining == 0:
                sc_remaining = int(rng.integers(3, 7))
                lap_time = base_pace * 1.40
                sc_count += 1
                second_sc_lap = None
            # Check for forced VSC
            elif forced_vsc_lap and lap == forced_vsc_lap and vsc_remaining == 0:
                vsc_remaining = int(rng.integers(2, 5))
                lap_time = base_pace * 1.20
                forced_vsc_lap = None
            # Stochastic SC (only if not suppressed)
            elif not suppress_sc:
                if rng.random() < sc_prob_per_lap and 1 < lap < total_laps - 3:
                    sc_remaining = int(rng.integers(3, 7))
                    lap_time = base_pace * 1.40
                    sc_count += 1
                elif rng.random() < vsc_prob_per_lap and 1 < lap < total_laps - 2:
                    vsc_remaining = int(rng.integers(2, 5))
                    lap_time = base_pace * 1.20
        
        # Pit stop
        if (current_stint < len(strategy.stints) - 1 and
                laps_in_stint >= stint.target_laps):
            lap_time += circuit.pit_loss_seconds + rng.normal(0, 0.5)
            current_stint += 1
            laps_in_stint = 0
        
        total_time += lap_time
    
    return total_time, sc_count


def run_scenario_mc(
    scenario: Scenario,
    strategy: Strategy,
    circuit: CircuitConfig,
    deg_model: xgb.XGBRegressor,
    feature_cols: list,
    fuel_config: dict,
    n_sims: int = 500,
    seed: int = 42,
) -> dict:
    """Run conditioned Monte Carlo for one scenario × one strategy."""
    rng = np.random.default_rng(seed)
    
    # Precompute deg rates (with multiplier applied inside simulation)
    deg_rates = precompute_deg_rates(strategy, circuit, deg_model, feature_cols)
    
    start_fuel = fuel_config["start_fuel_kg"]
    fuel_effect_val = fuel_config["fuel_effect_per_kg_seconds"]
    burn_rate = start_fuel / circuit.total_laps
    base_pace = 90.0
    
    if circuit.sc_prob_per_race > 0:
        sc_prob_per_lap = 1 - (1 - circuit.sc_prob_per_race) ** (1 / circuit.total_laps)
    else:
        sc_prob_per_lap = 0
    if circuit.vsc_prob_per_race > 0:
        vsc_prob_per_lap = 1 - (1 - circuit.vsc_prob_per_race) ** (1 / circuit.total_laps)
    else:
        vsc_prob_per_lap = 0
    
    # Determine second SC window for double_sc scenario
    total = circuit.total_laps
    second_sc_window = None
    if scenario.id == "double_sc":
        mid_start = int(total * 0.25) + 1
        mid_end = int(total * 0.60)
        second_sc_window = (mid_start, mid_end)
    
    total_times = []
    sc_counts = []
    
    for _ in range(n_sims):
        t, sc = simulate_race_conditioned(
            strategy, circuit, deg_rates, fuel_config,
            sc_prob_per_lap, vsc_prob_per_lap, base_pace,
            burn_rate, start_fuel, fuel_effect_val, rng,
            forced_sc_window=scenario.forced_sc_window,
            forced_vsc_window=scenario.forced_vsc_window,
            suppress_sc=scenario.suppress_sc,
            deg_multiplier=scenario.deg_multiplier,
            second_sc_window=second_sc_window,
        )
        total_times.append(t)
        sc_counts.append(sc)
    
    times = np.array(total_times)
    
    return {
        "strategy_name": strategy.name,
        "compound_sequence": strategy.compound_sequence,
        "num_stops": strategy.num_stops,
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "std_time": float(np.std(times)),
        "p5_time": float(np.percentile(times, 5)),
        "p95_time": float(np.percentile(times, 95)),
        "mean_sc_events": float(np.mean(sc_counts)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DECISION TRIGGER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_decision_triggers(
    scenario_results: list,
    default_best: str,
) -> list:
    """
    Extract actionable decision triggers from scenario analysis.
    
    Compares each scenario's best strategy to the default (unconditioned)
    plan and generates human-readable decision rules.
    """
    triggers = []
    
    for sr in scenario_results:
        scenario_best = sr["rankings"][0]["strategy_name"]
        scenario_best_stops = sr["rankings"][0]["num_stops"]
        default_rank = next(
            (i + 1 for i, r in enumerate(sr["rankings"]) if r["strategy_name"] == default_best),
            None
        )
        
        if scenario_best == default_best:
            continue  # same strategy — no trigger needed
        
        time_saved = sr["time_delta_vs_default"]
        
        triggers.append({
            "scenario_id": sr["scenario_id"],
            "scenario_name": sr["scenario_name"],
            "icon": sr["icon"],
            "probability": sr["probability"],
            "trigger": f"If {sr['scenario_name'].lower()}: switch to {_clean_name(scenario_best)}",
            "action": _clean_name(scenario_best),
            "action_stops": scenario_best_stops,
            "time_saved": round(time_saved, 1),
            "default_rank": default_rank,
        })
    
    # Sort by impact (time saved)
    triggers.sort(key=lambda x: x["time_saved"], reverse=True)
    
    return triggers


def _clean_name(raw: str) -> str:
    """Strip pit lap numbers: '1-stop MEDIUM→HARD (25/32)' -> '1-stop M→H'"""
    name = re.sub(r'\s*\(\d+(?:/\d+)*\)\s*', '', raw)
    name = name.replace('MEDIUM', 'M').replace('HARD', 'H').replace('SOFT', 'S')
    return name.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario_analysis(
    circuit_key: str,
    season: int,
    n_sims: int = 500,
    config_path: str = "configs/config.yaml",
) -> dict:
    """
    Run complete scenario analysis for a circuit.
    
    1. Load circuit config & degradation model
    2. Generate candidate strategies
    3. Run unconditioned MC to find default best
    4. Generate race scenarios
    5. Run conditioned MC for each scenario × strategy
    6. Extract decision triggers
    """
    config = load_config(config_path)
    raw_paths = config["paths"]["raw"]
    fuel_config = config["modeling"]["fuel_model"]
    
    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"
    
    logger.info("=" * 65)
    logger.info("  SCENARIO-BASED CONTINGENCY PLANNER")
    logger.info("=" * 65)
    
    # Load circuit
    circuit = load_circuit_config(circuit_key, season, circuit_csv, sc_priors_path, weather_dir)
    logger.info(f"  Circuit:  {circuit.circuit_name}")
    logger.info(f"  SC prob:  {circuit.sc_prob_per_race:.0%}")
    
    # Load model
    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    
    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]
    
    # Generate strategies
    strategies = generate_strategies(circuit)
    logger.info(f"  Strategies: {len(strategies)}")
    
    # ── Step 1: Unconditioned MC (find default best) ──
    logger.info("\n  Step 1: Unconditioned Monte Carlo (baseline)")
    from src.simulation.strategy_simulator import run_monte_carlo
    
    baseline_results = []
    for i, strategy in enumerate(strategies):
        result = run_monte_carlo(
            strategy, circuit, deg_model, feature_cols,
            fuel_config, n_sims=n_sims, seed=42 + i,
        )
        baseline_results.append(result)
    
    baseline_results.sort(key=lambda x: x["median_time"])
    default_best = baseline_results[0]
    default_best_name = default_best["strategy_name"]
    default_best_median = default_best["median_time"]
    
    logger.info(f"  Default best: {_clean_name(default_best_name)} "
                f"(median {default_best_median:.1f}s)")
    
    # ── Step 2: Generate scenarios ──
    scenarios = generate_scenarios(circuit)
    logger.info(f"\n  Step 2: Analysing {len(scenarios)} scenarios")
    
    # ── Step 3: Conditioned MC per scenario ──
    scenario_results = []
    t0 = time.time()
    
    for scenario in scenarios:
        logger.info(f"    {scenario.icon} {scenario.name}...")
        
        rankings = []
        for i, strategy in enumerate(strategies):
            result = run_scenario_mc(
                scenario, strategy, circuit, deg_model, feature_cols,
                fuel_config, n_sims=n_sims, seed=42 + i,
            )
            rankings.append(result)
        
        rankings.sort(key=lambda x: x["median_time"])
        best_median = rankings[0]["median_time"]
        
        # Calculate deltas relative to this scenario's best
        for j, r in enumerate(rankings):
            r["rank"] = j + 1
            r["delta"] = round(r["median_time"] - best_median, 1)
            r["clean_name"] = _clean_name(r["strategy_name"])
        
        # Find where the default plan ranks in this scenario
        default_in_scenario = next(
            (r for r in rankings if r["strategy_name"] == default_best_name),
            None
        )
        default_rank = default_in_scenario["rank"] if default_in_scenario else len(rankings)
        default_delta = default_in_scenario["delta"] if default_in_scenario else 0
        
        # Time saved by switching from default to scenario-optimal
        time_delta = round(default_delta, 1)
        
        scenario_results.append({
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "description": scenario.description,
            "icon": scenario.icon,
            "probability": scenario.base_probability,
            "rankings": rankings[:10],  # top 10 per scenario
            "scenario_best": _clean_name(rankings[0]["strategy_name"]),
            "scenario_best_stops": rankings[0]["num_stops"],
            "default_plan_rank": default_rank,
            "time_delta_vs_default": time_delta,
        })
        
        status = "✓ same" if default_rank == 1 else f"⚡ switch → {_clean_name(rankings[0]['strategy_name'])} (saves {time_delta:.1f}s)"
        logger.info(f"      Best: {_clean_name(rankings[0]['strategy_name'])} | Default: P{default_rank} | {status}")
    
    elapsed = time.time() - t0
    
    # ── Step 4: Extract decision triggers ──
    triggers = extract_decision_triggers(scenario_results, default_best_name)
    
    logger.info(f"\n  ✓ Scenario analysis complete in {elapsed:.1f}s")
    logger.info(f"  Decision triggers: {len(triggers)}")
    for t in triggers[:5]:
        logger.info(f"    {t['icon']} {t['trigger']} (saves {t['time_saved']}s)")
    
    # ── Build output ──
    output = {
        "circuit_key": circuit_key,
        "circuit_name": circuit.circuit_name,
        "season": season,
        "sc_probability": circuit.sc_prob_per_race,
        "n_sims_per_scenario": n_sims,
        "n_strategies": len(strategies),
        "elapsed_seconds": round(elapsed, 1),
        "default_plan": {
            "name": default_best_name,
            "clean_name": _clean_name(default_best_name),
            "compound_sequence": default_best["compound_sequence"],
            "num_stops": default_best["num_stops"],
            "median_time": default_best["median_time"],
        },
        "scenarios": scenario_results,
        "decision_triggers": triggers,
    }
    
    # Save
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    filename = f"scenarios_{circuit_key}_{season}.json"
    with open(output_dir / filename, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  ✓ Saved: results/{filename}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Scenario-based contingency planner")
    parser.add_argument("--circuit", type=str, help="Circuit key (e.g. bahrain)")
    parser.add_argument("--all", action="store_true", help="Run all circuits")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    if args.all:
        # Load all circuit keys
        config = load_config(args.config)
        raw_paths = config["paths"]["raw"]
        import pandas as pd
        circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
        circuits_df = pd.read_csv(circuit_csv)
        circuit_keys = circuits_df[circuits_df["season"] == args.season]["circuit_key"].unique()
        
        logger.info(f"Running scenario analysis for {len(circuit_keys)} circuits...")
        for key in sorted(circuit_keys):
            try:
                run_scenario_analysis(key, args.season, args.n_sims, args.config)
            except Exception as e:
                logger.error(f"  ✗ {key}: {e}")
    elif args.circuit:
        run_scenario_analysis(args.circuit, args.season, args.n_sims, args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
