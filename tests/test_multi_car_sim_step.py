"""Golden regression for the step-able refactor of multi_car_sim.
Needs numpy only. Run: python tests/test_multi_car_sim_step.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.multi_car_sim import (
    MultiCarRaceSim, DriverConfig, CircuitParams, Strategy,
)


def _scenario():
    drivers = [
        DriverConfig(f"D{i}", f"D{i}", "t" + str(i // 2), 0.05 * i, 0.5, 0.7, "")
        for i in range(6)
    ]
    circuit = CircuitParams(
        circuit_key="t", circuit_name="T", total_laps=30,
        pit_loss_seconds=22.0, sc_prob_per_race=0.4, vsc_prob_per_race=0.2,
        overtaking_difficulty=0.5, deg_rates={},
    )
    strat = Strategy(stints=[("MEDIUM", 15), ("HARD", 15)], name="MH")
    return circuit, drivers, [strat] * 6, strat


def test_run_is_deterministic_and_stable():
    circuit, drivers, strategies, strat = _scenario()
    r1 = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False).run(seed=123)
    r2 = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False).run(seed=123)
    assert r1["finishing_positions"] == r2["finishing_positions"]
    assert r1["target_time"] == r2["target_time"]
    assert sorted(r1["finishing_positions"]) == list(range(1, 7))


def test_reset_step_matches_run():
    circuit, drivers, strategies, strat = _scenario()
    ref = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False).run(seed=123)

    sim = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False)
    sim.reset(seed=123)
    while not sim.done:
        sim.step()
    res = sim.results()
    assert res["finishing_positions"] == ref["finishing_positions"]
    assert res["target_time"] == ref["target_time"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
