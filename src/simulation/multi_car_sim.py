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


from src.simulation.regulation_profiles import RegulationProfile, DEFAULT_PROFILE

# ── Constants ──────────────────────────────────────────────────
# Era-specific physics now live in RegulationProfile (see regulation_profiles.py).
# These module-level names are kept for backward compatibility (e.g.
# precompute_scenarios imports COMPOUND_DEG_BASE) and derive from the default
# (2022-25) profile so there is a single source of truth.
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]            # compound NAMES (era-independent)
COMPOUND_DEG_BASE = DEFAULT_PROFILE.compound_deg_base
COMPOUND_CLIFF = DEFAULT_PROFILE.compound_cliff


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
        profile: RegulationProfile = None,
    ):
        self.circuit = circuit
        self.drivers = drivers
        self.n_cars = len(drivers)
        self.strategies = strategies
        self.target_idx = target_driver_idx
        self.target_strategy = target_strategy
        self.greedy_sc = greedy_sc
        self.profile = profile if profile is not None else DEFAULT_PROFILE
        
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
        self.burn_rate = self.profile.start_fuel_kg / tl
    
    def _get_deg_rate(self, compound: str, driver: DriverConfig, circuit: CircuitParams) -> float:
        """Get degradation rate, scaled by driver tyre management."""
        base = circuit.deg_rates.get(compound, self.profile.compound_deg_base[compound])
        # Compound relativity: SOFT degrades faster, HARD slower (the XGBoost deg
        # model is compound-insensitive, so we inject the relative ordering here).
        base *= self.profile.compound_deg_multiplier.get(compound, 1.0)
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
            return self.profile.base_pace * self.profile.sc_pace_factor
        if vsc_active:
            return self.profile.base_pace * self.profile.vsc_pace_factor

        # Base + driver delta + compound pace offset (SOFT faster fresh)
        lap_time = self.profile.base_pace + driver.pace_delta
        lap_time += self.profile.compound_pace_offset.get(car.tyre_compound, 0.0)

        # Fuel effect
        fuel_remaining = max(0, self.profile.start_fuel_kg - self.burn_rate * (lap - 1))
        lap_time += fuel_remaining * self.profile.fuel_effect_per_kg

        # Tyre degradation (quadratic model matching RL env)
        deg_rate = self._get_deg_rate(car.tyre_compound, driver, self.circuit)
        tyre_deg = deg_rate * car.tyre_age + 0.002 * (car.tyre_age ** 1.3)
        lap_time += tyre_deg

        # Dirty air penalty (within the era's dirty-air window of car ahead)
        if dirty_air and gap_to_ahead < self.profile.dirty_air_window:
            lap_time += self.profile.dirty_air_penalty * (
                self.profile.dirty_air_window - gap_to_ahead
            ) / self.profile.dirty_air_window

        # Overtaking aid (DRS / 2026 override) within the era's window of car ahead
        if 0 < gap_to_ahead < self.profile.drs_window:
            drs_benefit = self.profile.overtake_aid_benefit * self.circuit.overtaking_difficulty
            lap_time -= drs_benefit

        # Random variation
        lap_time += rng.normal(0, self.profile.lap_time_noise_std)

        return max(lap_time, self.profile.base_pace * 0.95)  # floor
    
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
        cliff = self.profile.compound_cliff.get(car.tyre_compound, 30)
        tyre_urgency = car.tyre_age / cliff  # 0-1+, >0.5 means past halfway
        
        # Higher urgency = more likely to pit
        # Also pit if it's "free" under SC and we're past 40% of compound life
        return tyre_urgency > 0.4
    
    def _process_pit_stop(
        self, car: CarState, driver: DriverConfig,
        current_lap: int, sc_active: bool, vsc_active: bool,
        rng: np.random.Generator, forced_compound: str = None,
    ) -> float:
        """Execute pit stop. Returns time cost."""
        # Determine next compound
        if forced_compound is not None:
            next_compound = forced_compound
        elif car.next_pit_idx < len(car.strategy.stints) - 1:
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
    
    def reset(self, seed: int = 42):
        """Initialise a race; ready for step()."""
        self.rng = np.random.default_rng(seed)
        self.n_laps = self.circuit.total_laps
        self.cars = []
        for i, driver in enumerate(self.drivers):
            s = self.strategies[i] if i != self.target_idx else self.target_strategy
            car = CarState(driver_idx=i, tyre_compound=s.stints[0][0], strategy=s)
            car.compounds_used = {s.stints[0][0]}
            car.cumulative_time = i * 0.8
            self.cars.append(car)
        self.positions = self._update_positions(self.cars)
        self.position_history = np.zeros((self.n_laps, self.n_cars), dtype=int)
        self.sc_laps, self.vsc_laps, self.pit_events = [], [], []
        self.target_history = {"lap_times": [], "compounds": [], "tyre_ages": [],
                               "positions": [], "pit_laps": [], "sc_laps": []}
        self.sc_active = self.vsc_active = False
        self.sc_remaining = self.vsc_remaining = 0
        self.lap = 0
        self.done = False
        return self.positions

    def step(self, pit_override: dict = None):
        """Advance one lap. pit_override maps car_idx -> compound name (pit) or
        None (stay). Cars absent from the dict use their own strategy / greedy SC
        logic, which is how run() reproduces the original behaviour exactly."""
        lap = self.lap + 1
        self.lap = lap

        sc_just_started = False
        if self.sc_remaining > 0:
            self.sc_remaining -= 1
            self.sc_active = True
            self.sc_laps.append(lap)
        elif self.vsc_remaining > 0:
            self.vsc_remaining -= 1
            self.vsc_active = True
            self.vsc_laps.append(lap)
        else:
            self.sc_active = False
            self.vsc_active = False
            if self.rng.random() < self.sc_prob_per_lap and 1 < lap < self.n_laps - 3:
                self.sc_remaining = int(self.rng.integers(3, 7))
                self.sc_active = True
                sc_just_started = True
                self.sc_laps.append(lap)
            elif self.rng.random() < self.vsc_prob_per_lap and 1 < lap < self.n_laps - 2:
                self.vsc_remaining = int(self.rng.integers(2, 5))
                self.vsc_active = True
                self.vsc_laps.append(lap)

        if sc_just_started:
            self._compress_field_sc(self.cars, self.positions)

        gaps = self._compute_gaps(self.cars, self.positions)

        for i, (car, driver) in enumerate(zip(self.cars, self.drivers)):
            car.tyre_age += 1

            forced_compound = None
            if pit_override is not None and i in pit_override:
                chosen = pit_override[i]
                should_pit = (chosen is not None and car.stops_done < 3
                              and car.tyre_age >= 3 and lap < self.n_laps)
                forced_compound = chosen
            elif i == self.target_idx and self.greedy_sc:
                should_pit = self._should_pit_strategy(car, lap)
                if not should_pit and (self.sc_active or self.vsc_active):
                    should_pit = self._should_pit_greedy_sc(
                        car, lap, self.sc_active or self.vsc_active,
                        self.positions, gaps, self.rng)
            else:
                should_pit = self._should_pit_strategy(car, lap)

            pit_cost = 0.0
            if should_pit and lap < self.n_laps:
                pit_cost = self._process_pit_stop(
                    car, driver, lap, self.sc_active, self.vsc_active, self.rng,
                    forced_compound=forced_compound)
                self.pit_events.append({"lap": lap, "driver_idx": i,
                                        "compound": car.tyre_compound})
                if i == self.target_idx:
                    self.target_history["pit_laps"].append(lap)

            dirty_air = (gaps[i] < self.profile.dirty_air_window
                         and not self.sc_active and not self.vsc_active)
            lap_time = self._compute_lap_time(
                car, driver, lap, self.sc_active, self.vsc_active, self.rng,
                gap_to_ahead=gaps[i], dirty_air=dirty_air)
            lap_time += pit_cost
            car.cumulative_time += lap_time

            if i == self.target_idx:
                self.target_history["lap_times"].append(round(lap_time, 2))
                self.target_history["compounds"].append(car.tyre_compound)
                self.target_history["tyre_ages"].append(car.tyre_age)
                if lap in self.sc_laps:
                    self.target_history["sc_laps"].append(lap)

        self.positions = self._update_positions(self.cars)
        self.position_history[lap - 1] = self.positions
        self._check_lapped(self.cars, self.positions, lap)
        if not self.sc_active and not self.vsc_active:
            gaps = self._compute_gaps(self.cars, self.positions)
            self._process_overtaking(self.cars, self.positions, gaps, lap, self.rng)
            self.positions = self._update_positions(self.cars)
            self.position_history[lap - 1] = self.positions
        self.target_history["positions"].append(int(self.positions[self.target_idx]))

        if lap >= self.n_laps:
            self.done = True
        return self.positions

    def results(self) -> dict:
        """Assemble the result dict (same shape as the original run() return)."""
        final_positions = self.positions
        return {
            "finishing_positions": final_positions.tolist(),
            "position_history": self.position_history.tolist(),
            "target_position": int(final_positions[self.target_idx]),
            "target_time": round(self.cars[self.target_idx].cumulative_time, 1),
            "sc_laps": self.sc_laps,
            "vsc_laps": self.vsc_laps,
            "pit_events": self.pit_events,
            "target_history": self.target_history,
            "n_sc_events": len(set(self.sc_laps)),
        }

    def run(self, seed: int = 42) -> dict:
        """Full race = reset() then step() every lap. Behaviour identical to the
        original loop (same code path)."""
        self.reset(seed)
        while not self.done:
            self.step()
        return self.results()


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
