"""Build the per-car observation from sim state — shared by the training env and
the eval race runner so the policy sees identical inputs in both."""
from __future__ import annotations

import numpy as np

from src.rl.ma_obs import build_obs

_COMPOUND_IDX = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}


def car_obs(sim, circuit, profile, drivers, i: int) -> np.ndarray:
    car = sim.cars[i]
    driver = drivers[i]
    n_cars = len(drivers)
    gaps = sim._compute_gaps(sim.cars, sim.positions)
    order = list(np.argsort(sim.positions))
    pos_idx = order.index(i)
    gap_behind = gaps[order[pos_idx + 1]] if pos_idx + 1 < n_cars else 999.0
    state = dict(
        lap=sim.lap, total_laps=circuit.total_laps,
        tyre_age=car.tyre_age, compound_idx=_COMPOUND_IDX.get(car.tyre_compound, 1),
        cumulative_deg=0.0, position=int(sim.positions[i]), n_cars=n_cars,
        gap_ahead=float(gaps[i]), gap_behind=float(gap_behind),
        fuel_frac=max(0.0, 1.0 - sim.burn_rate * sim.lap / max(profile.start_fuel_kg, 1)),
        sc_active=int(sim.sc_active or sim.vsc_active),
        stops_done=car.stops_done, max_stops=3, compounds_used=len(car.compounds_used),
        laps_since_sc=0,
        driver_pace=min(driver.pace_delta, 2.0) / 2.0,
        driver_overtaking=driver.overtaking, driver_tyre=driver.tyre_management,
        pit_loss=circuit.pit_loss_seconds, sc_prob=circuit.sc_prob_per_race,
        overtaking_difficulty=circuit.overtaking_difficulty,
    )
    return build_obs(state)
