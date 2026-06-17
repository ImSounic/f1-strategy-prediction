"""Pure tests for multi-agent RL helpers. Run: python tests/test_ma_obs.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.ma_obs import (
    action_to_compound, legal_action_mask, reward_from_positions, build_obs, OBS_DIM,
    terminal_reward, DSQ_PENALTY,
)


def test_action_to_compound():
    assert action_to_compound(0) is None          # stay
    assert action_to_compound(1) == "SOFT"
    assert action_to_compound(2) == "MEDIUM"
    assert action_to_compound(3) == "HARD"


def test_legal_mask_stay_always_legal():
    m = legal_action_mask(stops_done=3, tyre_age=1, current_lap=1, total_laps=50)
    assert m[0] is True and m[1:] == [False, False, False]   # max stops + lap1 + young tyre


def test_legal_mask_allows_pit_midrace():
    m = legal_action_mask(stops_done=0, tyre_age=10, current_lap=20, total_laps=50)
    assert m == [True, True, True, True]


def test_legal_mask_blocks_late_and_young():
    assert legal_action_mask(0, 10, 49, 50)[1:] == [False, False, False]  # too late
    assert legal_action_mask(0, 2, 20, 50)[1:] == [False, False, False]   # tyre too young


def test_reward_positions_gained():
    assert reward_from_positions(grid=10, finish=4) == 6.0   # gained 6
    assert reward_from_positions(grid=1, finish=3) == -2.0   # lost 2


def test_terminal_reward_dsq_dominates():
    assert terminal_reward(5, 2, used_two_compounds=True) == 3.0      # legal: gained 3
    assert terminal_reward(1, 1, used_two_compounds=True) == 0.0
    assert terminal_reward(20, 1, used_two_compounds=False) == DSQ_PENALTY  # DSQ regardless of result
    # DSQ must be strictly worse than the worst legal finish (pole -> last, big field)
    assert DSQ_PENALTY < (1 - 22)


def test_build_obs_shape_and_bounds():
    state = dict(lap=10, total_laps=50, tyre_age=8, compound_idx=2, cumulative_deg=1.2,
                 position=5, n_cars=20, gap_ahead=1.3, gap_behind=0.8, fuel_frac=0.6,
                 sc_active=0, stops_done=1, max_stops=3, compounds_used=2, laps_since_sc=12,
                 driver_pace=0.3, driver_overtaking=0.6, driver_tyre=0.7,
                 pit_loss=23.0, sc_prob=0.55, overtaking_difficulty=0.5)
    obs = build_obs(state)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype.name == "float32"
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.5


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
