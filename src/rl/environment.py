"""
F1 Strategy RL Environment
============================
Gymnasium environment that wraps the existing Monte Carlo simulator.

The agent makes one decision per lap: PIT or STAY OUT.
When pitting, compound selection follows a deterministic rule
(hardest unused compound) to keep the action space minimal.

State space (14 continuous features, normalised to [0,1]):
    - Race progress, tyre age, compound info
    - Cumulative degradation, current pace delta
    - SC/VSC status, pit stop history
    - Circuit characteristics (pit cost, SC probability)

Action space: Discrete(2) — 0=stay, 1=pit

Reward: Shaped per-lap reward combining:
    - Negative normalised lap time (fast laps = good)
    - SC pit bonus (pitting under SC is nearly free)
    - Tyre cliff penalty (running past optimal stint length)
    - Compound rule penalty (must use 2+ compounds)

The environment uses the SAME physics as simulate_race_fast():
    - XGBoost degradation predictions
    - Quadratic tyre wear model
    - Fuel burn correction
    - Stochastic SC/VSC injection with Bayesian priors

Usage:
    env = F1StrategyEnv(circuit, deg_model, feature_cols, fuel_config)
    obs, info = env.reset()
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from src.simulation.strategy_simulator import (
    CircuitConfig,
    predict_degradation,
    COMPOUND_HARDNESS,
)


class F1StrategyEnv(gym.Env):
    """
    F1 race as a Markov Decision Process.
    
    Each step = 1 lap. Agent decides: pit (1) or stay out (0).
    Episode = 1 complete race (total_laps steps).
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Compound definitions: name, index, cliff threshold (laps)
    COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
    COMPOUND_IDX = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
    # Approximate cliff thresholds per compound (laps before severe drop-off)
    CLIFF_THRESHOLDS = {"HARD": 40, "MEDIUM": 30, "SOFT": 20}
    
    def __init__(
        self,
        circuit: CircuitConfig,
        deg_model,
        feature_cols: list,
        fuel_config: dict,
        max_stops: int = 3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.circuit = circuit
        self.deg_model = deg_model
        self.feature_cols = feature_cols
        self.fuel_config = fuel_config
        self.max_stops = max_stops
        self.render_mode = render_mode
        
        self.total_laps = circuit.total_laps
        self.base_pace = 90.0
        self.start_fuel = fuel_config["start_fuel_kg"]
        self.fuel_effect = fuel_config["fuel_effect_per_kg_seconds"]
        self.burn_rate = self.start_fuel / self.total_laps
        
        # SC probabilities (same formula as simulator)
        if circuit.sc_prob_per_race > 0:
            self.sc_prob_per_lap = 1 - (1 - circuit.sc_prob_per_race) ** (1 / self.total_laps)
        else:
            self.sc_prob_per_lap = 0.0
        if circuit.vsc_prob_per_race > 0:
            self.vsc_prob_per_lap = 1 - (1 - circuit.vsc_prob_per_race) ** (1 / self.total_laps)
        else:
            self.vsc_prob_per_lap = 0.0
        
        # Precompute degradation rates for all compounds × stint numbers
        self._precompute_deg_rates()
        
        # Gymnasium spaces
        self.action_space = spaces.Discrete(2)  # 0=stay, 1=pit
        
        # 14-dimensional state, normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )
    
    def _precompute_deg_rates(self):
        """Precompute degradation rates for all compound/stint combos."""
        self.deg_rates = {}
        for compound in self.COMPOUNDS:
            actual = getattr(self.circuit, f"{compound.lower()}_compound")
            hardness = COMPOUND_HARDNESS.get(actual, 3)
            for stint_num in range(1, self.max_stops + 2):
                rate = predict_degradation(
                    self.deg_model, hardness, stint_num,
                    25,  # reference stint length
                    self.circuit, self.feature_cols,
                )
                self.deg_rates[(compound, stint_num)] = float(max(0.005, min(rate, 0.5)))
    
    def _get_deg_rate(self, compound: str, stint_number: int) -> float:
        """Look up precomputed deg rate with noise."""
        key = (compound, min(stint_number, self.max_stops + 1))
        base = self.deg_rates.get(key, 0.05)
        noisy = base + self.rng.normal(0, 0.005)
        return max(0.005, noisy)
    
    def _select_next_compound(self) -> str:
        """Rule-based compound selection: hardest unused, then hardest available."""
        unused = [c for c in self.COMPOUNDS if c not in self.compounds_used]
        if unused:
            return unused[0]  # hardest unused (HARD > MEDIUM > SOFT)
        # All used — pick hardest available
        return "HARD"
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Race state
        self.current_lap = 0
        self.tyre_age = 0
        self.current_compound = "MEDIUM"  # start on MEDIUM (most common)
        self.compounds_used = {"MEDIUM"}
        self.stops_done = 0
        self.total_time = 0.0
        self.cumulative_deg = 0.0
        
        # SC state
        self.sc_active = False
        self.vsc_active = False
        self.sc_remaining = 0
        self.vsc_remaining = 0
        self.laps_since_sc = 99
        self.total_sc_events = 0
        
        # Current stint deg rate
        self.current_deg_rate = self._get_deg_rate(self.current_compound, 1)
        
        # History for replay/visualization
        self.history = {
            "lap_times": [],
            "compounds": [],
            "tyre_ages": [],
            "pit_laps": [],
            "sc_laps": [],
            "vsc_laps": [],
            "actions": [],
            "rewards": [],
        }
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self.current_lap += 1
        self.tyre_age += 1
        
        # ── Process pit stop action ──
        pitted = False
        pit_cost = 0.0
        
        # Can only pit if: action=1, haven't exceeded max stops, 
        # tyre age >= 3 (no instant re-pits), not final lap
        if (action == 1 and 
            self.stops_done < self.max_stops and 
            self.tyre_age >= 3 and 
            self.current_lap < self.total_laps):
            
            pitted = True
            next_compound = self._select_next_compound()
            
            # Pit time loss (reduced under SC/VSC)
            # Under SC: field bunches but pit lane time is unchanged (~10-12s loss)
            # Under VSC: smaller benefit (~14-16s loss)
            if self.sc_active:
                pit_cost = self.circuit.pit_loss_seconds * 0.50  # ~11.5s for typical 23s pit
            elif self.vsc_active:
                pit_cost = self.circuit.pit_loss_seconds * 0.65  # ~15s for typical 23s pit
            else:
                pit_cost = self.circuit.pit_loss_seconds + self.rng.normal(0, 0.5)
            
            # Reset tyre state
            self.current_compound = next_compound
            self.compounds_used.add(next_compound)
            self.stops_done += 1
            self.tyre_age = 1  # fresh tyres
            self.current_deg_rate = self._get_deg_rate(
                self.current_compound, self.stops_done + 1
            )
            self.cumulative_deg = 0.0
            
            self.history["pit_laps"].append(self.current_lap)
        
        # ── Compute lap time ──
        
        # Fuel effect
        fuel_remaining = max(0, self.start_fuel - self.burn_rate * (self.current_lap - 1))
        fuel_time = fuel_remaining * self.fuel_effect
        
        # Tyre degradation (same model as simulator)
        tyre_deg = self.current_deg_rate * self.tyre_age + 0.002 * (self.tyre_age ** 1.3)
        self.cumulative_deg = tyre_deg
        
        # ── SC/VSC events ──
        was_sc = self.sc_active
        was_vsc = self.vsc_active
        
        if self.sc_remaining > 0:
            self.sc_remaining -= 1
            self.sc_active = True
            lap_time = self.base_pace * 1.40
            self.history["sc_laps"].append(self.current_lap)
            self.laps_since_sc = 0
        elif self.vsc_remaining > 0:
            self.vsc_remaining -= 1
            self.vsc_active = True
            lap_time = self.base_pace * 1.20
            self.history["vsc_laps"].append(self.current_lap)
            self.laps_since_sc = 0
        else:
            self.sc_active = False
            self.vsc_active = False
            self.laps_since_sc += 1
            
            # Normal lap time
            lap_time = self.base_pace + fuel_time + tyre_deg + self.rng.normal(0, 0.3)
            
            # Check for new SC/VSC (same logic as simulator)
            if (self.rng.random() < self.sc_prob_per_lap and 
                1 < self.current_lap < self.total_laps - 3):
                self.sc_remaining = int(self.rng.integers(3, 7))
                lap_time = self.base_pace * 1.40
                self.sc_active = True
                self.total_sc_events += 1
                self.history["sc_laps"].append(self.current_lap)
                self.laps_since_sc = 0
            elif (self.rng.random() < self.vsc_prob_per_lap and
                  1 < self.current_lap < self.total_laps - 2):
                self.vsc_remaining = int(self.rng.integers(2, 5))
                lap_time = self.base_pace * 1.20
                self.vsc_active = True
                self.history["vsc_laps"].append(self.current_lap)
                self.laps_since_sc = 0
        
        # Add pit stop cost
        lap_time += pit_cost
        
        self.total_time += lap_time
        
        # ── Reward ──
        reward = self._compute_reward(lap_time, pitted, pit_cost)
        
        # ── Record history ──
        self.history["lap_times"].append(lap_time)
        self.history["compounds"].append(self.current_compound)
        self.history["tyre_ages"].append(self.tyre_age)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        
        # ── Termination ──
        terminated = self.current_lap >= self.total_laps
        truncated = False
        
        # Terminal bonus/penalty
        if terminated:
            # Must have used 2+ compounds (FIA rule)
            if len(self.compounds_used) < 2:
                reward -= 50.0  # severe penalty — illegal strategy
            
            # DOMINANT TERMINAL REWARD: total race time vs reference
            # Reference = base_pace × total_laps (theoretical minimum)
            reference_time = self.base_pace * self.total_laps
            # Scale: each second saved/lost → ~0.1 reward units
            # Total reward from this term: roughly -30 to -50 range
            # This ensures the agent optimises for TOTAL TIME, not per-lap shaping
            time_ratio = self.total_time / reference_time
            reward += -(time_ratio - 1.0) * 100.0  # 1% slower = -1.0 reward
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _compute_reward(self, lap_time: float, pitted: bool, pit_cost: float) -> float:
        """
        Per-lap reward with minimal shaping + strong terminal signal.
        
        Design rationale:
            - Primary signal: total race time (terminal reward)
            - Per-lap shaping kept minimal to avoid over-pitting
            - SC pit bonus preserved (key reactive behavior)
            - NO cliff penalty — degradation model already makes old-tyre
              laps slower, which the agent learns from naturally
        
        v2 changes from v1:
            - Removed cliff penalty (was causing 2.67 mean stops instead of ~1.5)
            - Reduced per-lap time penalty scale (10.0 → 20.0 divisor)
            - Added significant pit cost penalty (proportional to actual time lost)
            - Terminal reward dominates: -total_time / reference_time
        """
        reward = 0.0
        
        # 1. Small per-lap time penalty (gentle shaping, not dominant)
        if not (self.sc_active or self.vsc_active):
            # Normalise: 0 at base_pace, negative for slower laps
            reward = -(lap_time - self.base_pace) / 20.0
        else:
            reward = 0.0  # SC laps are out of driver's control
        
        # 2. SC pit bonus — the KEY reactive behavior we want
        if pitted and self.sc_active:
            reward += 2.3  # bonus scaled to ~11.5s savings (50% vs 100%)
        elif pitted and self.vsc_active:
            reward += 1.4  # bonus scaled to ~8s savings (65% vs 100%)
        
        # 3. Pit cost penalty — proportional to actual time lost
        # This discourages unnecessary stops by reflecting real cost
        if pitted:
            reward -= pit_cost / 10.0  # ~2.3 penalty for normal stop, ~0.8 under SC
        
        return float(reward)
    
    def _get_obs(self) -> np.ndarray:
        """Build 14-dimensional normalised observation."""
        compound_idx = self.COMPOUND_IDX.get(self.current_compound, 1)
        cliff = self.CLIFF_THRESHOLDS.get(self.current_compound, 30)
        
        obs = np.array([
            # Race progress
            self.current_lap / self.total_laps,
            # Tyre state
            self.tyre_age / 50.0,
            (compound_idx + 1) / 3.0,
            self.current_deg_rate / 0.2,  # normalise typical range [0, 0.2]
            min(self.cumulative_deg, 10.0) / 10.0,
            self.tyre_age / max(cliff, 1),  # proximity to cliff (1.0 = at cliff)
            # Strategy state
            self.stops_done / self.max_stops,
            len(self.compounds_used) / 3.0,
            # SC state
            float(self.sc_active or self.vsc_active),
            min(self.laps_since_sc, 20) / 20.0,
            # Fuel state
            max(0, self.start_fuel - self.burn_rate * self.current_lap) / self.start_fuel,
            # Circuit characteristics (static but helps generalisation)
            self.circuit.pit_loss_seconds / 30.0,
            self.circuit.sc_prob_per_race,
            # Remaining laps
            max(0, self.total_laps - self.current_lap) / self.total_laps,
        ], dtype=np.float32)
        
        return np.clip(obs, 0.0, 1.5)  # allow slight overflow but cap at 1.5
    
    def _get_info(self) -> dict:
        """Return info dict for logging/evaluation."""
        return {
            "total_time": self.total_time,
            "current_lap": self.current_lap,
            "stops_done": self.stops_done,
            "compounds_used": list(self.compounds_used),
            "current_compound": self.current_compound,
            "tyre_age": self.tyre_age,
            "sc_events": self.total_sc_events,
            "history": self.history,
        }