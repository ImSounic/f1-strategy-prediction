"""
Monte Carlo Race Strategy Simulator
=====================================
Simulates full race strategies by combining:
    - Tyre degradation model (XGBoost) for lap time prediction
    - Bayesian SC priors for stochastic safety car injection
    - Pit stop loss model from circuit data
    - Fuel correction model

For each strategy (compound sequence + pit lap windows), runs N Monte Carlo
simulations with randomised SC events and degradation noise to produce
a distribution of total race times.

Core concepts:
    Strategy = sequence of (compound, target_stint_length) tuples
    Simulation = one full race with stochastic events
    Result = distribution of total race times across N simulations

Output:
    Per-strategy: mean, median, std, P5/P95 of total race time
    Optimal strategy recommendation with confidence interval

Usage:
    python -m src.simulation.strategy_simulator --circuit bahrain --season 2024
    python -m src.simulation.strategy_simulator --circuit monza --season 2024 --n-sims 2000
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitConfig:
    """All parameters for a specific circuit."""
    circuit_key: str
    circuit_name: str
    total_laps: int
    pit_loss_seconds: float
    circuit_length_km: float
    # Pirelli characteristics
    asphalt_abrasiveness: float
    asphalt_grip: float
    traction_demand: float
    braking_severity: float
    lateral_forces: float
    tyre_stress: float
    downforce_level: float
    track_evolution: float
    # Compound allocation
    hard_compound: str
    medium_compound: str
    soft_compound: str
    # SC probabilities (Bayesian)
    sc_prob_per_race: float
    vsc_prob_per_race: float
    # Weather defaults
    mean_track_temp: float = 35.0
    mean_air_temp: float = 25.0
    mean_humidity: float = 50.0
    mean_wind_speed: float = 2.0
    track_temp_range: float = 5.0


@dataclass
class StintPlan:
    """One stint in a strategy."""
    compound: str          # "SOFT", "MEDIUM", "HARD"
    actual_compound: str   # "C1", "C2", etc.
    compound_hardness: int # 1-6
    target_laps: int       # planned stint length


@dataclass
class Strategy:
    """Complete race strategy = sequence of stints."""
    name: str
    stints: list  # list of StintPlan
    
    @property
    def total_laps(self):
        return sum(s.target_laps for s in self.stints)
    
    @property
    def num_stops(self):
        return len(self.stints) - 1
    
    @property
    def compound_sequence(self):
        return " → ".join(s.compound for s in self.stints)


@dataclass
class SimulationResult:
    """Result of one Monte Carlo simulation."""
    total_time: float
    lap_times: list
    sc_laps: list
    vsc_laps: list
    pit_laps: list
    stint_summary: list


# ═══════════════════════════════════════════════════════════════════════════
#  CIRCUIT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def load_circuit_config(
    circuit_key: str,
    season: int,
    circuit_csv: Path,
    sc_priors_path: Path,
    weather_dir: Path = None,
) -> CircuitConfig:
    """Load all circuit parameters from our data."""
    circuits = pd.read_csv(circuit_csv)
    
    row = circuits[
        (circuits["circuit_key"] == circuit_key) &
        (circuits["season"] == season)
    ]
    
    if row.empty:
        # Try matching by name
        row = circuits[
            circuits["circuit_name"].str.lower().str.contains(circuit_key.lower()) &
            (circuits["season"] == season)
        ]
    
    if row.empty:
        available = circuits[circuits["season"] == season]["circuit_key"].tolist()
        raise ValueError(
            f"Circuit '{circuit_key}' not found for {season}. "
            f"Available: {available}"
        )
    
    row = row.iloc[0]
    
    # SC priors
    with open(sc_priors_path) as f:
        sc_priors = json.load(f)
    
    circuit_sc = sc_priors.get(row["circuit_key"], {})
    sc_prob = circuit_sc.get("bayesian_sc_prob", 0.55)
    vsc_prob = circuit_sc.get("bayesian_vsc_prob", 0.45)
    
    # Weather defaults from historical data
    mean_track_temp = 35.0
    mean_air_temp = 25.0
    mean_humidity = 50.0
    mean_wind_speed = 2.0
    track_temp_range = 5.0
    
    if weather_dir and weather_dir.exists():
        weather_files = list(weather_dir.glob(f"{season}_*_{row['circuit_name'].replace(' ', '_')}*.parquet"))
        if not weather_files:
            # Try broader match
            for f in weather_dir.glob(f"{season}_*.parquet"):
                df = pd.read_parquet(f)
                if df["RoundNumber"].iloc[0] == row["round_number"]:
                    weather_files = [f]
                    break
        
        if weather_files:
            wdf = pd.read_parquet(weather_files[0])
            mean_track_temp = float(wdf["TrackTemp"].mean())
            mean_air_temp = float(wdf["AirTemp"].mean())
            mean_humidity = float(wdf["Humidity"].mean())
            mean_wind_speed = float(wdf["WindSpeed"].mean())
            track_temp_range = float(wdf["TrackTemp"].max() - wdf["TrackTemp"].min())
    
    return CircuitConfig(
        circuit_key=row["circuit_key"],
        circuit_name=row["circuit_name"],
        total_laps=int(row["total_laps"]),
        pit_loss_seconds=float(row["pit_loss_seconds"]),
        circuit_length_km=float(row["circuit_length_km"]),
        asphalt_abrasiveness=float(row["asphalt_abrasiveness"]),
        asphalt_grip=float(row["asphalt_grip"]),
        traction_demand=float(row["traction_demand"]),
        braking_severity=float(row["braking_severity"]),
        lateral_forces=float(row["lateral_forces"]),
        tyre_stress=float(row["tyre_stress"]),
        downforce_level=float(row["downforce_level"]),
        track_evolution=float(row["track_evolution"]),
        hard_compound=row["hard_compound"],
        medium_compound=row["medium_compound"],
        soft_compound=row["soft_compound"],
        sc_prob_per_race=sc_prob,
        vsc_prob_per_race=vsc_prob,
        mean_track_temp=mean_track_temp,
        mean_air_temp=mean_air_temp,
        mean_humidity=mean_humidity,
        mean_wind_speed=mean_wind_speed,
        track_temp_range=track_temp_range,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY GENERATION
# ═══════════════════════════════════════════════════════════════════════════

COMPOUND_HARDNESS = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6}


def generate_strategies(circuit: CircuitConfig) -> list:
    """
    Generate candidate strategies for a given circuit.
    
    Rules (FIA regulations):
        - Must use at least 2 different dry compounds during the race
        - Common strategies: 1-stop, 2-stop, (rarely 3-stop)
    """
    strategies = []
    total = circuit.total_laps
    
    H, M, S = circuit.hard_compound, circuit.medium_compound, circuit.soft_compound
    Hh = COMPOUND_HARDNESS[H]
    Mh = COMPOUND_HARDNESS[M]
    Sh = COMPOUND_HARDNESS[S]
    
    # ── 1-Stop Strategies ──
    one_stop_splits = [
        (0.45, 0.55), (0.50, 0.50), (0.55, 0.45),
        (0.40, 0.60), (0.60, 0.40),
    ]
    
    compound_pairs_1stop = [
        ("MEDIUM", M, Mh, "HARD", H, Hh),
        ("HARD", H, Hh, "MEDIUM", M, Mh),
        ("SOFT", S, Sh, "HARD", H, Hh),
        ("SOFT", S, Sh, "MEDIUM", M, Mh),
        ("MEDIUM", M, Mh, "SOFT", S, Sh),
        ("HARD", H, Hh, "SOFT", S, Sh),
    ]
    
    for split in one_stop_splits:
        for name1, c1, h1, name2, c2, h2 in compound_pairs_1stop:
            laps1 = max(5, int(total * split[0]))
            laps2 = total - laps1
            if laps2 < 5:
                continue
            
            strategy = Strategy(
                name=f"1-stop {name1}→{name2} ({laps1}/{laps2})",
                stints=[
                    StintPlan(name1, c1, h1, laps1),
                    StintPlan(name2, c2, h2, laps2),
                ],
            )
            strategies.append(strategy)
    
    # ── 2-Stop Strategies ──
    two_stop_splits = [
        (0.30, 0.35, 0.35), (0.25, 0.35, 0.40),
        (0.33, 0.33, 0.34), (0.25, 0.40, 0.35),
        (0.20, 0.40, 0.40), (0.35, 0.30, 0.35),
    ]
    
    compound_triples = [
        ("SOFT", S, Sh, "MEDIUM", M, Mh, "HARD", H, Hh),
        ("SOFT", S, Sh, "HARD", H, Hh, "MEDIUM", M, Mh),
        ("MEDIUM", M, Mh, "HARD", H, Hh, "SOFT", S, Sh),
        ("MEDIUM", M, Mh, "SOFT", S, Sh, "HARD", H, Hh),
        ("HARD", H, Hh, "MEDIUM", M, Mh, "SOFT", S, Sh),
        ("MEDIUM", M, Mh, "HARD", H, Hh, "MEDIUM", M, Mh),
        ("SOFT", S, Sh, "MEDIUM", M, Mh, "MEDIUM", M, Mh),
        ("SOFT", S, Sh, "HARD", H, Hh, "HARD", H, Hh),
    ]
    
    for split in two_stop_splits:
        for n1, c1, h1, n2, c2, h2, n3, c3, h3 in compound_triples:
            laps1 = max(5, int(total * split[0]))
            laps2 = max(5, int(total * split[1]))
            laps3 = total - laps1 - laps2
            if laps3 < 5:
                continue
            
            # Check 2-compound rule
            compounds_used = {n1, n2, n3}
            if len(compounds_used) < 2:
                continue
            
            strategy = Strategy(
                name=f"2-stop {n1}→{n2}→{n3} ({laps1}/{laps2}/{laps3})",
                stints=[
                    StintPlan(n1, c1, h1, laps1),
                    StintPlan(n2, c2, h2, laps2),
                    StintPlan(n3, c3, h3, laps3),
                ],
            )
            strategies.append(strategy)
    
    # Deduplicate by name
    seen = set()
    unique = []
    for s in strategies:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)
    
    logger.info(f"  Generated {len(unique)} candidate strategies "
                f"({sum(1 for s in unique if s.num_stops == 1)} 1-stop, "
                f"{sum(1 for s in unique if s.num_stops == 2)} 2-stop)")
    
    return unique


# ═══════════════════════════════════════════════════════════════════════════
#  DEGRADATION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_degradation(
    model: xgb.XGBRegressor,
    compound_hardness: int,
    stint_number: int,
    stint_length: int,
    circuit: CircuitConfig,
    feature_cols: list,
) -> float:
    """
    Predict degradation rate (DegSlope) for a given stint configuration.
    Returns predicted DegSlope in seconds/lap.
    """
    features = {
        "CompoundHardness": compound_hardness,
        "StintNumber": stint_number,
        "StintLength": stint_length,
        "TyreLifeStart": 0,
        "asphalt_abrasiveness": circuit.asphalt_abrasiveness,
        "asphalt_grip": circuit.asphalt_grip,
        "traction_demand": circuit.traction_demand,
        "braking_severity": circuit.braking_severity,
        "lateral_forces": circuit.lateral_forces,
        "tyre_stress": circuit.tyre_stress,
        "downforce_level": circuit.downforce_level,
        "track_evolution": circuit.track_evolution,
        "circuit_length_km": circuit.circuit_length_km,
        "pit_loss_seconds": circuit.pit_loss_seconds,
        "MeanTrackTemp": circuit.mean_track_temp,
        "MeanAirTemp": circuit.mean_air_temp,
        "MeanHumidity": circuit.mean_humidity,
        "MeanWindSpeed": circuit.mean_wind_speed,
        "TrackTempRange": circuit.track_temp_range,
    }
    
    # Build feature vector in correct order
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    
    deg_slope = model.predict(X)[0]
    
    # Clamp to physical bounds (can't have negative degradation realistically)
    deg_slope = max(0.005, min(deg_slope, 0.5))
    
    return float(deg_slope)


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE RACE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_race(
    strategy: Strategy,
    circuit: CircuitConfig,
    deg_model: xgb.XGBRegressor,
    feature_cols: list,
    fuel_config: dict,
    rng: np.random.Generator,
) -> SimulationResult:
    """
    Simulate one complete race with stochastic events.
    
    Lap time model:
        lap_time = base_pace + fuel_effect + tyre_degradation + noise
    
    Stochastic events:
        - Safety Car: drawn per-lap from Bernoulli with circuit SC rate
          SC lasts 3-6 laps, lap time = 120% of base (slow pace)
        - VSC: similar, lap time = 140% of base (very slow)
        - Lap time noise: N(0, 0.3s) representing traffic, driver variation
    """
    total_laps = circuit.total_laps
    start_fuel = fuel_config["start_fuel_kg"]
    fuel_effect = fuel_config["fuel_effect_per_kg_seconds"]
    
    # Per-race burn rate
    burn_rate = start_fuel / total_laps
    
    # Base pace: approximate from historical (we use a reference value,
    # the simulator cares about *relative* times between strategies)
    base_pace = 90.0  # reference pace in seconds (doesn't affect strategy ranking)
    
    # SC probability per lap (convert race probability to per-lap)
    # P(SC in race) = 1 - (1 - p_lap)^total_laps
    # → p_lap = 1 - (1 - P_race)^(1/total_laps)
    if circuit.sc_prob_per_race > 0:
        sc_prob_per_lap = 1 - (1 - circuit.sc_prob_per_race) ** (1 / total_laps)
    else:
        sc_prob_per_lap = 0
    
    if circuit.vsc_prob_per_race > 0:
        vsc_prob_per_lap = 1 - (1 - circuit.vsc_prob_per_race) ** (1 / total_laps)
    else:
        vsc_prob_per_lap = 0
    
    # Predict degradation rate per stint
    stint_deg_rates = []
    for i, stint in enumerate(strategy.stints):
        deg_rate = predict_degradation(
            deg_model, stint.compound_hardness, i + 1,
            stint.target_laps, circuit, feature_cols,
        )
        # Add noise to degradation prediction
        deg_rate += rng.normal(0, 0.01)
        deg_rate = max(0.005, deg_rate)
        stint_deg_rates.append(deg_rate)
    
    # Simulate lap by lap
    lap_times = []
    sc_laps = []
    vsc_laps = []
    pit_laps = []
    stint_summary = []
    
    current_stint = 0
    laps_in_stint = 0
    sc_remaining = 0
    vsc_remaining = 0
    
    for lap in range(1, total_laps + 1):
        stint = strategy.stints[current_stint]
        laps_in_stint += 1
        
        # Fuel effect
        fuel_remaining = max(0, start_fuel - burn_rate * (lap - 1))
        fuel_time = fuel_remaining * fuel_effect
        
        # Tyre degradation (quadratic model: slope increases with age)
        deg_rate = stint_deg_rates[current_stint]
        tyre_deg = deg_rate * laps_in_stint + 0.002 * (laps_in_stint ** 1.3)
        
        # Check for SC/VSC (not during existing SC/VSC)
        if sc_remaining > 0:
            # Under Safety Car: fixed slow pace
            lap_time = base_pace * 1.40  # ~40% slower under SC
            sc_remaining -= 1
            sc_laps.append(lap)
        elif vsc_remaining > 0:
            # Under VSC: moderately slow
            lap_time = base_pace * 1.20
            vsc_remaining -= 1
            vsc_laps.append(lap)
        else:
            # Normal racing lap
            lap_time = base_pace + fuel_time + tyre_deg
            
            # Random noise (traffic, battles, mistakes)
            lap_time += rng.normal(0, 0.3)
            
            # Check for new SC/VSC event
            if rng.random() < sc_prob_per_lap and lap > 1 and lap < total_laps - 3:
                sc_remaining = rng.integers(3, 7)  # SC lasts 3-6 laps
                lap_time = base_pace * 1.40
                sc_laps.append(lap)
            elif rng.random() < vsc_prob_per_lap and lap > 1 and lap < total_laps - 2:
                vsc_remaining = rng.integers(2, 5)  # VSC lasts 2-4 laps
                lap_time = base_pace * 1.20
                vsc_laps.append(lap)
        
        lap_times.append(lap_time)
        
        # Pit stop at end of stint
        if (current_stint < len(strategy.stints) - 1 and
                laps_in_stint >= stint.target_laps):
            # Pit stop time loss (reduced under SC/VSC)
            # Under SC: field bunches but pit lane time unchanged (~10-12s loss)
            # Under VSC: smaller benefit (~14-16s loss)
            if sc_remaining > 0:
                pit_time = circuit.pit_loss_seconds * 0.50  # ~11.5s for typical 23s pit
            elif vsc_remaining > 0:
                pit_time = circuit.pit_loss_seconds * 0.65  # ~15s for typical 23s pit
            else:
                pit_time = circuit.pit_loss_seconds + rng.normal(0, 0.5)
            lap_times[-1] += pit_time
            pit_laps.append(lap)
            
            stint_summary.append({
                "stint": current_stint + 1,
                "compound": stint.compound,
                "laps": laps_in_stint,
                "deg_rate": deg_rate,
            })
            
            current_stint += 1
            laps_in_stint = 0
    
    # Final stint summary
    stint_summary.append({
        "stint": current_stint + 1,
        "compound": strategy.stints[current_stint].compound,
        "laps": laps_in_stint,
        "deg_rate": stint_deg_rates[current_stint],
    })
    
    return SimulationResult(
        total_time=sum(lap_times),
        lap_times=lap_times,
        sc_laps=sc_laps,
        vsc_laps=vsc_laps,
        pit_laps=pit_laps,
        stint_summary=stint_summary,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MONTE CARLO ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def precompute_deg_rates(
    strategy: Strategy,
    circuit: CircuitConfig,
    deg_model: xgb.XGBRegressor,
    feature_cols: list,
) -> list:
    """Precompute degradation rates once per strategy (avoid repeated predict calls)."""
    rates = []
    for i, stint in enumerate(strategy.stints):
        deg_rate = predict_degradation(
            deg_model, stint.compound_hardness, i + 1,
            stint.target_laps, circuit, feature_cols,
        )
        rates.append(float(max(0.005, min(deg_rate, 0.5))))
    return rates


def simulate_race_fast(
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
) -> tuple:
    """Optimised simulation — no model calls, pure numpy."""
    total_laps = circuit.total_laps
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
        tyre_deg = deg_rate * laps_in_stint + 0.002 * (laps_in_stint ** 1.3)

        if sc_remaining > 0:
            lap_time = base_pace * 1.40
            sc_remaining -= 1
        elif vsc_remaining > 0:
            lap_time = base_pace * 1.20
            vsc_remaining -= 1
        else:
            lap_time = base_pace + fuel_time + tyre_deg + rng.normal(0, 0.3)

            if rng.random() < sc_prob_per_lap and 1 < lap < total_laps - 3:
                sc_remaining = rng.integers(3, 7)
                lap_time = base_pace * 1.40
                sc_count += 1
            elif rng.random() < vsc_prob_per_lap and 1 < lap < total_laps - 2:
                vsc_remaining = rng.integers(2, 5)
                lap_time = base_pace * 1.20

        # Pit stop
        if (current_stint < len(strategy.stints) - 1 and
                laps_in_stint >= stint.target_laps):
            lap_time += circuit.pit_loss_seconds + rng.normal(0, 0.5)
            current_stint += 1
            laps_in_stint = 0

        total_time += lap_time

    return total_time, sc_count


def run_monte_carlo(
    strategy: Strategy,
    circuit: CircuitConfig,
    deg_model: xgb.XGBRegressor,
    feature_cols: list,
    fuel_config: dict,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict:
    """Run N simulations with precomputed degradation rates."""
    rng = np.random.default_rng(seed)

    # Precompute once
    deg_rates = precompute_deg_rates(strategy, circuit, deg_model, feature_cols)

    start_fuel = fuel_config["start_fuel_kg"]
    fuel_effect = fuel_config["fuel_effect_per_kg_seconds"]
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

    total_times = []
    sc_counts = []

    for _ in range(n_sims):
        t, sc = simulate_race_fast(
            strategy, circuit, deg_rates, fuel_config,
            sc_prob_per_lap, vsc_prob_per_lap, base_pace,
            burn_rate, start_fuel, fuel_effect, rng,
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
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "mean_sc_events": float(np.mean(sc_counts)),
        "n_sims": n_sims,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_simulator(
    circuit_key: str,
    season: int,
    n_sims: int = 1000,
    config_path: str = "configs/config.yaml",
):
    """Run the full strategy simulation pipeline."""
    config = load_config(config_path)
    
    raw_paths = config["paths"]["raw"]
    fuel_config = config["modeling"]["fuel_model"]
    
    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"
    
    logger.info("=" * 65)
    logger.info("  MONTE CARLO STRATEGY SIMULATOR")
    logger.info("=" * 65)
    
    # Load circuit
    circuit = load_circuit_config(circuit_key, season, circuit_csv, sc_priors_path, weather_dir)
    logger.info(f"  Circuit:    {circuit.circuit_name} ({season})")
    logger.info(f"  Laps:       {circuit.total_laps}")
    logger.info(f"  Pit loss:   {circuit.pit_loss_seconds}s")
    logger.info(f"  Compounds:  H={circuit.hard_compound} M={circuit.medium_compound} S={circuit.soft_compound}")
    logger.info(f"  SC prob:    {circuit.sc_prob_per_race:.0%}")
    logger.info(f"  Track temp: {circuit.mean_track_temp:.1f}°C")
    
    # Load degradation model
    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    
    # Load feature columns
    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]
    
    # Generate strategies
    strategies = generate_strategies(circuit)
    
    # Run Monte Carlo for each strategy
    logger.info(f"\n  Running {n_sims} simulations per strategy...")
    t0 = time.time()
    
    all_results = []
    for i, strategy in enumerate(strategies):
        result = run_monte_carlo(
            strategy, circuit, deg_model, feature_cols,
            fuel_config, n_sims=n_sims, seed=42 + i,
        )
        all_results.append(result)
    
    elapsed = time.time() - t0
    total_sims = len(strategies) * n_sims
    logger.info(f"  Completed {total_sims:,} simulations in {elapsed:.1f}s "
                f"({total_sims/elapsed:.0f} sims/s)")
    
    # Sort by median time (most robust ranking)
    all_results.sort(key=lambda x: x["median_time"])
    
    # Display results
    logger.info("\n" + "═" * 65)
    logger.info("  STRATEGY RANKINGS (by median total race time)")
    logger.info("═" * 65)
    
    best_median = all_results[0]["median_time"]
    
    logger.info(f"  {'Rank':<5} {'Strategy':<45} {'Median':>8} {'Delta':>7} {'Std':>6}")
    logger.info(f"  {'─'*5} {'─'*45} {'─'*8} {'─'*7} {'─'*6}")
    
    for rank, res in enumerate(all_results[:20], 1):
        delta = res["median_time"] - best_median
        delta_str = f"+{delta:.2f}s" if delta > 0 else "BEST"
        logger.info(
            f"  {rank:<5} {res['strategy_name']:<45} "
            f"{res['median_time']:>8.2f} {delta_str:>7} {res['std_time']:>5.2f}s"
        )
    
    # Detailed top 5
    logger.info("\n" + "─" * 65)
    logger.info("  TOP 5 DETAILED ANALYSIS")
    logger.info("─" * 65)
    
    for rank, res in enumerate(all_results[:5], 1):
        logger.info(f"\n  #{rank} {res['strategy_name']}")
        logger.info(f"     Compounds: {res['compound_sequence']}")
        logger.info(f"     Stops:     {res['num_stops']}")
        logger.info(f"     Median:    {res['median_time']:.2f}s")
        logger.info(f"     Mean:      {res['mean_time']:.2f}s ± {res['std_time']:.2f}s")
        logger.info(f"     Range:     [{res['p5_time']:.2f}, {res['p95_time']:.2f}] (P5-P95)")
        logger.info(f"     SC events: {res['mean_sc_events']:.1f} avg per race")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output = {
        "circuit": {
            "name": circuit.circuit_name,
            "key": circuit.circuit_key,
            "season": season,
            "total_laps": circuit.total_laps,
            "pit_loss": circuit.pit_loss_seconds,
            "compounds": f"H={circuit.hard_compound} M={circuit.medium_compound} S={circuit.soft_compound}",
            "sc_probability": circuit.sc_prob_per_race,
            "track_temp": circuit.mean_track_temp,
        },
        "simulation": {
            "n_strategies": len(strategies),
            "n_sims_per_strategy": n_sims,
            "total_simulations": total_sims,
            "elapsed_seconds": elapsed,
        },
        "rankings": all_results,
    }
    
    filename = f"strategy_{circuit.circuit_key}_{season}.json"
    with open(output_dir / filename, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n  ✓ Results saved: results/{filename}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo race strategy simulator.")
    parser.add_argument("--circuit", type=str, required=True, help="Circuit key or name fragment")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_simulator(args.circuit, args.season, args.n_sims, args.config)


if __name__ == "__main__":
    main()
