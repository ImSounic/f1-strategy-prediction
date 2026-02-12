"""
Multi-Car F1 Race Simulator
=============================
Simulates a full 20-car field lap by lap with:
  - Individual driver pace offsets & tyre management
  - Position tracking via cumulative time
  - Overtaking model (DRS, dirty air, circuit difficulty)
  - Blue flags for lapped cars
  - Team orders between teammates
  - SC/VSC field compression
  - Pit stop position cost (real-world gap-based)

Used by the pre-compute pipeline to evaluate strategies in a full-field context.

Physics match the existing single-car simulator:
  lap_time = base_pace + driver_delta + fuel_effect + tyre_deg + noise
  - Quadratic tyre wear: deg_rate * age + 0.002 * age^1.3
  - Fuel burn: linear depletion, fuel_effect_per_kg * remaining_kg
  - SC pace: base_pace * 1.40 (full SC) or * 1.20 (VSC)
  - Pit cost reduced under SC (35%) or VSC (60%)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ──────────────────────────────────────────────────

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_DEG_BASE = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04}
COMPOUND_CLIFF = {"SOFT": 20, "MEDIUM": 30, "HARD": 40}

# Base pace reference (same as RL env)
BASE_PACE = 90.0
START_FUEL_KG = 110.0
FUEL_EFFECT_PER_KG = 0.035
SC_PACE_FACTOR = 1.40
VSC_PACE_FACTOR = 1.20
PIT_STOP_STATIONARY = 2.5  # seconds stationary (added on top of pit_loss)


@dataclass
class DriverConfig:
    """Per-driver configuration."""
    code: str
    name: str
    team: str
    pace_delta: float         # seconds slower than reference per lap
    overtaking: float         # 0-1, ability to overtake
    tyre_management: float    # 0-1, lower = less degradation
    teammate_code: str = ""


@dataclass
class CircuitParams:
    """Circuit parameters for the multi-car sim."""
    circuit_key: str
    circuit_name: str
    total_laps: int
    pit_loss_seconds: float
    sc_prob_per_race: float
    vsc_prob_per_race: float
    overtaking_difficulty: float  # 0-1, higher = easier to overtake
    # Compound-specific deg rates (from XGBoost model or defaults)
    deg_rates: dict = field(default_factory=dict)  # {compound: rate}


@dataclass  
class Strategy:
    """A race strategy: sequence of (compound, planned_laps) stints."""
    stints: list  # [(compound, n_laps), ...]
    name: str = ""
    
    @property
    def num_stops(self) -> int:
        return len(self.stints) - 1
    
    @property
    def pit_laps(self) -> list:
        """Compute pit lap numbers from stint lengths."""
        laps = []
        cumulative = 0
        for comp, n in self.stints[:-1]:
            cumulative += n
            laps.append(cumulative)
        return laps
    
    @property
    def compound_sequence(self) -> list:
        return [c for c, _ in self.stints]


@dataclass
class CarState:
    """Per-car mutable state during simulation."""
    driver_idx: int
    cumulative_time: float = 0.0
    position: int = 0
    tyre_compound: str = "MEDIUM"
    tyre_age: int = 0
    stint_number: int = 1
    stops_done: int = 0
    compounds_used: set = field(default_factory=lambda: {"MEDIUM"})
    lapped: bool = False
    lap_delta_to_leader: float = 0.0
    
    # Strategy tracking
    next_pit_idx: int = 0  # index into strategy.pit_laps
    strategy: Strategy = None


def generate_common_strategies(total_laps: int) -> list:
    """Generate candidate strategies (1-stop and 2-stop) for a given race length."""
    strategies = []
    
    # 1-stop strategies
    for first_compound in COMPOUNDS:
        for second_compound in COMPOUNDS:
            if first_compound == second_compound:
                continue
            for pit_frac in [0.35, 0.45, 0.55, 0.65]:
                pit_lap = max(5, min(total_laps - 5, int(total_laps * pit_frac)))
                s = Strategy(
                    stints=[(first_compound, pit_lap), 
                            (second_compound, total_laps - pit_lap)],
                    name=f"{first_compound[0]}{second_compound[0]} L{pit_lap}",
                )
                strategies.append(s)
    
    # 2-stop strategies
    for c1 in COMPOUNDS:
        for c2 in COMPOUNDS:
            for c3 in COMPOUNDS:
                if len({c1, c2, c3}) < 2:
                    continue
                for f1 in [0.28, 0.35]:
                    for f2 in [0.55, 0.65]:
                        p1 = max(5, min(total_laps - 10, int(total_laps * f1)))
                        p2 = max(p1 + 5, min(total_laps - 5, int(total_laps * f2)))
                        s = Strategy(
                            stints=[
                                (c1, p1),
                                (c2, p2 - p1),
                                (c3, total_laps - p2),
                            ],
                            name=f"{c1[0]}{c2[0]}{c3[0]} L{p1}/L{p2}",
                        )
                        strategies.append(s)
    
    return strategies


class MultiCarRaceSim:
    """
    Full 20-car race simulator.
    
    Usage:
        sim = MultiCarRaceSim(circuit, drivers, strategies, target_driver_idx)
        result = sim.run(seed=42)
    """
    
    def __init__(
        self,
        circuit: CircuitParams,
        drivers: list[DriverConfig],
        strategies: list[Strategy],  # one per driver, same order
        target_driver_idx: int,
        target_strategy: Strategy,
        greedy_sc: bool = True,
    ):
        self.circuit = circuit
        self.drivers = drivers
        self.n_cars = len(drivers)
        self.strategies = strategies
        self.target_idx = target_driver_idx
        self.target_strategy = target_strategy
        self.greedy_sc = greedy_sc
        
        # SC probabilities per lap
        tl = circuit.total_laps
        self.sc_prob_per_lap = (
            1 - (1 - circuit.sc_prob_per_race) ** (1 / tl)
            if circuit.sc_prob_per_race > 0 else 0.0
        )
        self.vsc_prob_per_lap = (
            1 - (1 - circuit.vsc_prob_per_race) ** (1 / tl)
            if circuit.vsc_prob_per_race > 0 else 0.0
        )
        
        # Fuel
        self.burn_rate = START_FUEL_KG / tl
    
    def _get_deg_rate(self, compound: str, driver: DriverConfig, circuit: CircuitParams) -> float:
        """Get degradation rate, scaled by driver tyre management."""
        base = circuit.deg_rates.get(compound, COMPOUND_DEG_BASE[compound])
        # Better tyre management (higher rating) = lower deg
        driver_factor = 1.0 + 0.3 * (1.0 - driver.tyre_management)
        return base * driver_factor
    
    def _compute_lap_time(
        self, 
        car: CarState, 
        driver: DriverConfig,
        lap: int, 
        sc_active: bool, 
        vsc_active: bool,
        rng: np.random.Generator,
        gap_to_ahead: float,
        dirty_air: bool,
    ) -> float:
        """Compute lap time for a single car."""
        
        if sc_active:
            return BASE_PACE * SC_PACE_FACTOR
        if vsc_active:
            return BASE_PACE * VSC_PACE_FACTOR
        
        # Base + driver delta
        lap_time = BASE_PACE + driver.pace_delta
        
        # Fuel effect
        fuel_remaining = max(0, START_FUEL_KG - self.burn_rate * (lap - 1))
        lap_time += fuel_remaining * FUEL_EFFECT_PER_KG
        
        # Tyre degradation (quadratic model matching RL env)
        deg_rate = self._get_deg_rate(car.tyre_compound, driver, self.circuit)
        tyre_deg = deg_rate * car.tyre_age + 0.002 * (car.tyre_age ** 1.3)
        lap_time += tyre_deg
        
        # Dirty air penalty (within 1.5s of car ahead)
        if dirty_air and gap_to_ahead < 1.5:
            lap_time += 0.15 * (1.5 - gap_to_ahead) / 1.5
        
        # DRS benefit (within 1.0s of car ahead)
        if gap_to_ahead > 0 and gap_to_ahead < 1.0:
            drs_benefit = 0.3 * self.circuit.overtaking_difficulty
            lap_time -= drs_benefit
        
        # Random variation
        lap_time += rng.normal(0, 0.3)
        
        return max(lap_time, BASE_PACE * 0.95)  # floor
    
    def _should_pit_strategy(self, car: CarState, current_lap: int) -> bool:
        """Check if car should pit according to its fixed strategy."""
        pit_laps = car.strategy.pit_laps
        if car.next_pit_idx < len(pit_laps):
            return current_lap == pit_laps[car.next_pit_idx]
        return False
    
    def _should_pit_greedy_sc(
        self, car: CarState, current_lap: int, 
        sc_active: bool, positions: np.ndarray,
        gaps: np.ndarray, rng: np.random.Generator,
    ) -> bool:
        """
        Greedy SC reactor for the target driver.
        Decides whether to pit under SC/VSC considering position impact.
        """
        if not (sc_active and current_lap < self.circuit.total_laps - 2):
            return False
        
        # Don't re-pit if tyres are fresh (< 5 laps old)
        if car.tyre_age < 5:
            return False
        
        # Don't pit if we've already done max stops
        if car.stops_done >= 3:
            return False
        
        # Count how many cars around us are also pitting under SC
        # Heuristic: pit if tyre age is above threshold
        cliff = COMPOUND_CLIFF.get(car.tyre_compound, 30)
        tyre_urgency = car.tyre_age / cliff  # 0-1+, >0.5 means past halfway
        
        # Higher urgency = more likely to pit
        # Also pit if it's "free" under SC and we're past 40% of compound life
        return tyre_urgency > 0.4
    
    def _process_pit_stop(
        self, car: CarState, driver: DriverConfig, 
        current_lap: int, sc_active: bool, vsc_active: bool,
        rng: np.random.Generator,
    ) -> float:
        """Execute pit stop. Returns time cost."""
        # Determine next compound
        if car.next_pit_idx < len(car.strategy.stints) - 1:
            next_compound = car.strategy.stints[car.next_pit_idx + 1][0]
        else:
            # Emergency/SC pit — pick hardest unused
            unused = [c for c in COMPOUNDS if c not in car.compounds_used]
            if unused:
                next_compound = unused[-1]  # HARD first
            else:
                next_compound = "HARD"
        
        # Time cost
        # Under SC: field bunches but pit lane time is unchanged (~10-12s loss)
        # Under VSC: smaller benefit (~14-16s loss)
        if sc_active:
            pit_cost = self.circuit.pit_loss_seconds * 0.50  # ~11.5s for typical 23s pit
        elif vsc_active:
            pit_cost = self.circuit.pit_loss_seconds * 0.65  # ~15s for typical 23s pit
        else:
            pit_cost = self.circuit.pit_loss_seconds + rng.normal(0, 0.5)
        
        # Update car state
        car.tyre_compound = next_compound
        car.compounds_used.add(next_compound)
        car.tyre_age = 0
        car.stint_number += 1
        car.stops_done += 1
        car.next_pit_idx += 1
        
        return max(pit_cost, 0.0)
    
    def _update_positions(self, cars: list[CarState]) -> np.ndarray:
        """Compute positions from cumulative times. Returns position array."""
        times = np.array([c.cumulative_time for c in cars])
        order = np.argsort(times)
        positions = np.empty(self.n_cars, dtype=int)
        for pos, car_idx in enumerate(order):
            positions[car_idx] = pos + 1  # 1-indexed
            cars[car_idx].position = pos + 1
        return positions
    
    def _compute_gaps(self, cars: list[CarState], positions: np.ndarray) -> np.ndarray:
        """Compute gap to car ahead for each car. Leader gets gap=999."""
        n = self.n_cars
        times = np.array([c.cumulative_time for c in cars])
        gaps = np.full(n, 999.0)
        
        # Sort by position
        order = np.argsort(positions)
        for i in range(1, n):
            car_idx = order[i]
            ahead_idx = order[i - 1]
            gaps[car_idx] = times[car_idx] - times[ahead_idx]
        
        return gaps
    
    def _process_overtaking(
        self, cars: list[CarState], positions: np.ndarray, 
        gaps: np.ndarray, lap: int, rng: np.random.Generator,
    ):
        """
        Process overtaking attempts.
        A car can overtake the car ahead if:
          - Gap < overtake_threshold
          - Random check based on driver skill and circuit difficulty
        Costs both cars time (fighting).
        """
        order = np.argsort(positions)
        swaps = []
        
        for i in range(1, self.n_cars):
            car_idx = order[i]
            ahead_idx = order[i - 1]
            car = cars[car_idx]
            ahead = cars[ahead_idx]
            driver = self.drivers[car_idx]
            driver_ahead = self.drivers[ahead_idx]
            
            gap = gaps[car_idx]
            if gap > 1.5 or gap <= 0:
                continue
            
            # Blue flags: if car ahead is lapped, let through (no cost)
            if ahead.lapped:
                # Auto-pass
                swaps.append((car_idx, ahead_idx, 0.0, 0.0))
                continue
            
            # Team orders: teammate behind with better pace yields
            if (driver.team == driver_ahead.team and 
                driver.pace_delta < driver_ahead.pace_delta and
                gap < 1.5):
                # Teammate yields — small time cost
                swaps.append((car_idx, ahead_idx, 0.0, 0.3))
                continue
            
            # Normal overtake attempt
            # Probability based on: gap, driver skill, tyre advantage, circuit
            tyre_adv = max(0, ahead.tyre_age - car.tyre_age) / 10.0  # 0-3+
            pace_adv = max(0, driver_ahead.pace_delta - driver.pace_delta)
            
            overtake_prob = (
                self.circuit.overtaking_difficulty * 0.3  # circuit base
                + driver.overtaking * 0.2                  # driver skill
                + min(tyre_adv, 1.0) * 0.3                # tyre advantage
                + min(pace_adv, 0.5) * 0.2                # pace advantage
            )
            
            # Scale by gap (closer = more likely)
            overtake_prob *= (1.5 - gap) / 1.5
            
            if rng.random() < overtake_prob:
                # Successful overtake — both lose time from fighting
                swaps.append((car_idx, ahead_idx, 0.3, 0.5))
        
        # Apply swaps (time penalties)
        for overtaker, defender, cost_overtaker, cost_defender in swaps:
            cars[overtaker].cumulative_time += cost_overtaker
            cars[defender].cumulative_time += cost_defender
    
    def _compress_field_sc(self, cars: list[CarState], positions: np.ndarray):
        """Safety car compresses gaps to ~1s between consecutive cars."""
        order = np.argsort(positions)
        leader_time = cars[order[0]].cumulative_time
        for i in range(1, self.n_cars):
            car_idx = order[i]
            target_time = leader_time + i * 1.0  # 1s gaps
            if cars[car_idx].cumulative_time > target_time + 0.5:
                cars[car_idx].cumulative_time = target_time
    
    def _check_lapped(self, cars: list[CarState], positions: np.ndarray, lap: int):
        """Mark cars that are a full lap behind as lapped."""
        if lap < 3:
            return
        order = np.argsort(positions)
        leader_time = cars[order[0]].cumulative_time
        lap_time_approx = leader_time / max(lap, 1)
        
        for car in cars:
            car.lap_delta_to_leader = (car.cumulative_time - leader_time) / max(lap_time_approx, 80)
            car.lapped = car.lap_delta_to_leader > 0.95  # nearly a full lap behind
    
    def run(self, seed: int = 42) -> dict:
        """
        Run a full race simulation.
        
        Returns dict with:
          - finishing_positions: array of final positions (1-indexed)
          - position_history: (n_laps, n_cars) array of positions per lap
          - target_position: final position of target driver
          - sc_laps: list of SC lap numbers
          - pit_events: list of {lap, driver_idx, compound} for all pit stops
          - target_history: detailed lap-by-lap for target driver
        """
        rng = np.random.default_rng(seed)
        n_laps = self.circuit.total_laps
        
        # ── Initialise cars ──
        cars = []
        for i, driver in enumerate(self.drivers):
            s = self.strategies[i] if i != self.target_idx else self.target_strategy
            car = CarState(
                driver_idx=i,
                tyre_compound=s.stints[0][0],
                strategy=s,
            )
            car.compounds_used = {s.stints[0][0]}
            # Starting gap based on grid position (0.8s per position)
            car.cumulative_time = i * 0.8
            cars.append(car)
        
        # Sort cars by initial grid order (already in order by index for now)
        positions = self._update_positions(cars)
        
        # ── Tracking ──
        position_history = np.zeros((n_laps, self.n_cars), dtype=int)
        sc_laps = []
        vsc_laps = []
        pit_events = []
        target_history = {
            "lap_times": [],
            "compounds": [],
            "tyre_ages": [],
            "positions": [],
            "pit_laps": [],
            "sc_laps": [],
        }
        
        sc_active = False
        vsc_active = False
        sc_remaining = 0
        vsc_remaining = 0
        sc_just_started = False
        
        for lap in range(1, n_laps + 1):
            # ── SC/VSC state ──
            sc_just_started = False
            if sc_remaining > 0:
                sc_remaining -= 1
                sc_active = True
                sc_laps.append(lap)
            elif vsc_remaining > 0:
                vsc_remaining -= 1
                vsc_active = True
                vsc_laps.append(lap)
            else:
                sc_active = False
                vsc_active = False
                
                # Check for new SC/VSC
                if (rng.random() < self.sc_prob_per_lap and 
                    1 < lap < n_laps - 3):
                    sc_remaining = int(rng.integers(3, 7))
                    sc_active = True
                    sc_just_started = True
                    sc_laps.append(lap)
                elif (rng.random() < self.vsc_prob_per_lap and
                      1 < lap < n_laps - 2):
                    vsc_remaining = int(rng.integers(2, 5))
                    vsc_active = True
                    vsc_laps.append(lap)
            
            # ── Compress field under SC ──
            if sc_just_started:
                self._compress_field_sc(cars, positions)
            
            # ── Compute gaps ──
            gaps = self._compute_gaps(cars, positions)
            
            # ── Process each car ──
            for i, (car, driver) in enumerate(zip(cars, self.drivers)):
                car.tyre_age += 1
                
                # ── Pit stop decision ──
                should_pit = False
                
                if i == self.target_idx and self.greedy_sc:
                    # Target driver: strategy + greedy SC reaction
                    should_pit = self._should_pit_strategy(car, lap)
                    if not should_pit and (sc_active or vsc_active):
                        should_pit = self._should_pit_greedy_sc(
                            car, lap, sc_active or vsc_active, positions, gaps, rng
                        )
                else:
                    # AI drivers: follow fixed strategy
                    should_pit = self._should_pit_strategy(car, lap)
                
                pit_cost = 0.0
                if should_pit and lap < n_laps:
                    pit_cost = self._process_pit_stop(
                        car, driver, lap, sc_active, vsc_active, rng
                    )
                    pit_events.append({
                        "lap": lap, "driver_idx": i, 
                        "compound": car.tyre_compound,
                    })
                    if i == self.target_idx:
                        target_history["pit_laps"].append(lap)
                
                # ── Lap time ──
                dirty_air = gaps[i] < 1.5 and not sc_active and not vsc_active
                lap_time = self._compute_lap_time(
                    car, driver, lap, sc_active, vsc_active, rng,
                    gap_to_ahead=gaps[i], dirty_air=dirty_air,
                )
                lap_time += pit_cost
                car.cumulative_time += lap_time
                
                # ── Target driver tracking ──
                if i == self.target_idx:
                    target_history["lap_times"].append(round(lap_time, 2))
                    target_history["compounds"].append(car.tyre_compound)
                    target_history["tyre_ages"].append(car.tyre_age)
                    if lap in sc_laps:
                        target_history["sc_laps"].append(lap)
            
            # ── Update positions ──
            positions = self._update_positions(cars)
            position_history[lap - 1] = positions
            
            # ── Check lapped cars ──
            self._check_lapped(cars, positions, lap)
            
            # ── Overtaking (not under SC) ──
            if not sc_active and not vsc_active:
                gaps = self._compute_gaps(cars, positions)
                self._process_overtaking(cars, positions, gaps, lap, rng)
                positions = self._update_positions(cars)
                position_history[lap - 1] = positions
            
            # Record target position
            target_history["positions"].append(int(positions[self.target_idx]))
        
        # ── Final results ──
        final_positions = positions
        target_pos = int(final_positions[self.target_idx])
        
        return {
            "finishing_positions": final_positions.tolist(),
            "position_history": position_history.tolist(),
            "target_position": target_pos,
            "target_time": round(cars[self.target_idx].cumulative_time, 1),
            "sc_laps": sc_laps,
            "vsc_laps": vsc_laps,
            "pit_events": pit_events,
            "target_history": target_history,
            "n_sc_events": len(set(sc_laps)),
        }


def build_grid(
    drivers: list[DriverConfig],
    target_driver_idx: int,
    target_grid_position: int,
) -> list[DriverConfig]:
    """
    Reorder drivers list so target is at target_grid_position.
    Other drivers fill remaining positions sorted by pace (best first).
    Returns new list where index = grid position - 1.
    """
    others = [d for i, d in enumerate(drivers) if i != target_driver_idx]
    # Sort others by pace (fastest first)
    others.sort(key=lambda d: d.pace_delta)
    
    # Build grid
    grid = []
    target = drivers[target_driver_idx]
    inserted = False
    other_idx = 0
    
    for pos in range(len(drivers)):
        if pos == target_grid_position - 1:
            grid.append(target)
            inserted = True
        else:
            if other_idx < len(others):
                grid.append(others[other_idx])
                other_idx += 1
    
    if not inserted:
        grid.append(target)
    
    return grid


def find_target_in_grid(grid: list[DriverConfig], target_code: str) -> int:
    """Find index of target driver in grid."""
    for i, d in enumerate(grid):
        if d.code == target_code:
            return i
    raise ValueError(f"Driver {target_code} not found in grid")
