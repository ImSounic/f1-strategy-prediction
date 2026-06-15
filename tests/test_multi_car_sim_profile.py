"""Behaviour-preservation tests for the profile refactor of multi_car_sim.
Needs numpy only. Run: python tests/test_multi_car_sim_profile.py"""
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.multi_car_sim import (
    MultiCarRaceSim, DriverConfig, CircuitParams, Strategy,
)
from src.simulation.regulation_profiles import GROUND_EFFECT_2022_25


def _scenario():
    drivers = [
        DriverConfig("AAA", "A", "t1", 0.0, 0.6, 0.7, "BBB"),
        DriverConfig("BBB", "B", "t1", 0.2, 0.5, 0.6, "AAA"),
        DriverConfig("CCC", "C", "t2", 0.4, 0.5, 0.6, "DDD"),
    ]
    circuit = CircuitParams(
        circuit_key="test", circuit_name="Test", total_laps=20,
        pit_loss_seconds=20.0, sc_prob_per_race=0.0, vsc_prob_per_race=0.0,
        overtaking_difficulty=0.5, deg_rates={},
    )
    strat = Strategy(stints=[("MEDIUM", 10), ("HARD", 10)], name="MH")
    return circuit, drivers, [strat, strat, strat], strat


def test_default_profile_matches_explicit_2022_25():
    circuit, drivers, strategies, strat = _scenario()
    a = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=42)
    b = MultiCarRaceSim(circuit, drivers, strategies, 0, strat,
                        profile=GROUND_EFFECT_2022_25).run(seed=42)
    assert a["target_time"] == b["target_time"]
    assert a["finishing_positions"] == b["finishing_positions"]


def test_deterministic_same_seed():
    circuit, drivers, strategies, strat = _scenario()
    a = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=7)
    b = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=7)
    assert a["target_time"] == b["target_time"]
    assert a["finishing_positions"] == b["finishing_positions"]


def test_sim_reads_profile_constants():
    # Lowering base_pace in the profile MUST lower simulated time -> proves the
    # sim reads the profile rather than stale module-level constants.
    circuit, drivers, strategies, strat = _scenario()
    base = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=42)
    faster = replace(GROUND_EFFECT_2022_25, base_pace=80.0)
    out = MultiCarRaceSim(circuit, drivers, strategies, 0, strat,
                          profile=faster).run(seed=42)
    assert out["target_time"] < base["target_time"]


def test_finishing_positions_are_a_permutation():
    circuit, drivers, strategies, strat = _scenario()
    r = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=1)
    assert sorted(r["finishing_positions"]) == [1, 2, 3]


def _solo_time(compound, laps, seed=42):
    drivers = [DriverConfig("AAA", "A", "t1", 0.0, 0.5, 0.7, "")]
    circuit = CircuitParams(
        circuit_key="t", circuit_name="T", total_laps=laps,
        pit_loss_seconds=20.0, sc_prob_per_race=0.0, vsc_prob_per_race=0.0,
        overtaking_difficulty=0.5, deg_rates={},
    )
    strat = Strategy(stints=[(compound, laps)], name=compound)
    return MultiCarRaceSim(circuit, drivers, [strat], 0, strat,
                           greedy_sc=False).run(seed=seed)["target_time"]


def test_soft_faster_when_fresh_short_race():
    # Over a short stint the pace offset dominates -> SOFT quicker than HARD.
    assert _solo_time("SOFT", 8) < _solo_time("HARD", 8)


def test_hard_faster_over_long_stint():
    # Over a long stint degradation dominates -> HARD quicker than SOFT.
    assert _solo_time("HARD", 45) < _solo_time("SOFT", 45)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
