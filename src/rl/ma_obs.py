"""
Pure helpers for the multi-agent F1 RL environment
==================================================
No Ray / no gym / no sim imports — unit-testable on any machine. The RLlib env
(multiagent_env.py) composes these to build observations, mask illegal actions,
translate actions to pit decisions, and compute rewards.
"""
from __future__ import annotations

import numpy as np

# Discrete(4) action -> compound to switch to (None = stay out)
_ACTION_COMPOUND = {0: None, 1: "SOFT", 2: "MEDIUM", 3: "HARD"}

MIN_STINT_LAPS = 3   # no re-pit before the tyre has this many laps
OBS_DIM = 18


def action_to_compound(action: int):
    """Map a Discrete(4) action to a compound name, or None for 'stay out'."""
    return _ACTION_COMPOUND[int(action)]


def legal_action_mask(stops_done: int, tyre_age: int, current_lap: int,
                      total_laps: int, max_stops: int = 3,
                      min_stint: int = MIN_STINT_LAPS) -> list:
    """Boolean mask over [stay, pit-S, pit-M, pit-H]. Stay is always legal.

    Pitting is illegal if max stops reached, the tyre is too young, or it's the
    last lap or two.
    """
    can_pit = (stops_done < max_stops
               and tyre_age >= min_stint
               and current_lap < total_laps - 1)
    return [True, can_pit, can_pit, can_pit]


def reward_from_positions(grid: int, finish: int) -> float:
    """Terminal reward = positions gained (grid - finish). Higher is better."""
    return float(grid - finish)


# Dry-race two-compound rule is enforced by DISQUALIFICATION, not a soft penalty.
# A DSQ removes the car from classification, so it must be strictly worse than any
# legal finish (worst legal positions-gained ~ -(field size)). -50 dominates.
DSQ_PENALTY = -50.0


def terminal_reward(grid: int, finish: int, used_two_compounds: bool = True) -> float:
    """Terminal reward: positions gained if legal, else a DSQ-level penalty
    (dry race run on a single compound -> disqualified)."""
    if not used_two_compounds:
        return DSQ_PENALTY
    return reward_from_positions(grid, finish)


def build_obs(state: dict) -> np.ndarray:
    """Build the normalised, driver-conditioned observation vector (OBS_DIM,)."""
    obs = np.array([
        state["lap"] / max(state["total_laps"], 1),
        min(state["tyre_age"], 50) / 50.0,
        (state["compound_idx"] + 1) / 3.0,
        min(state["cumulative_deg"], 10.0) / 10.0,
        state["position"] / max(state["n_cars"], 1),
        min(state["gap_ahead"], 5.0) / 5.0,
        min(state["gap_behind"], 5.0) / 5.0,
        state["fuel_frac"],
        float(state["sc_active"]),
        state["stops_done"] / max(state["max_stops"], 1),
        state["compounds_used"] / 3.0,
        min(state["laps_since_sc"], 20) / 20.0,
        state["driver_pace"],
        state["driver_overtaking"],
        state["driver_tyre"],
        state["pit_loss"] / 30.0,
        state["sc_prob"],
        state["overtaking_difficulty"],
    ], dtype=np.float32)
    return np.clip(obs, 0.0, 1.5)
