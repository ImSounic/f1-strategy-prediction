"""Run one evaluation race over the validated multi_car_sim with pluggable controllers."""
from __future__ import annotations

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy
from src.rl.ma_obs import action_to_compound, legal_action_mask
from src.rl.eval.obs_builder import car_obs


def run_race(circuit, drivers, controllers, profile, seed: int):
    """controllers: list aligned with drivers; each has .start_compound and .decide(obs).
    Returns finishing position per car index (lower = better)."""
    n = len(drivers)
    # single-stint strategies (no auto-pit); pits are driven entirely by controllers.
    strategies = [Strategy(stints=[(controllers[i].start_compound, circuit.total_laps)],
                           name=f"car_{i}") for i in range(n)]
    default_strat = Strategy(stints=[("MEDIUM", circuit.total_laps)], name="x")
    sim = MultiCarRaceSim(circuit, drivers, strategies, 0, default_strat,
                          greedy_sc=False, profile=profile)
    sim.reset(seed=seed)
    while not sim.done:
        pit_override = {}
        for i in range(n):
            obs = car_obs(sim, circuit, profile, drivers, i)
            act = int(controllers[i].decide(obs))
            mask = legal_action_mask(sim.cars[i].stops_done, sim.cars[i].tyre_age,
                                     sim.lap + 1, circuit.total_laps)
            pit_override[i] = action_to_compound(act) if mask[act] else None
        sim.step(pit_override)
    return [int(p) for p in sim.positions]
