"""Pure tests for driver-config helpers. Run: python tests/test_driver_config_helpers.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preparation.driver_config_helpers import (
    parse_laptime, best_quali_time, minmax_normalize, constructor_to_team,
    driver_name_from_id,
)


def test_parse_laptime_mmss():
    assert abs(parse_laptime("1:31.471") - 91.471) < 1e-6
    assert abs(parse_laptime("0:59.999") - 59.999) < 1e-6


def test_parse_laptime_plain_and_blank():
    assert abs(parse_laptime("88.123") - 88.123) < 1e-6
    assert parse_laptime("") is None
    assert parse_laptime(None) is None


def test_best_quali_time_session_priority():
    assert abs(best_quali_time("1:32.0", "1:31.0", "1:30.5") - 90.5) < 1e-6
    assert abs(best_quali_time("1:32.0", "1:31.0", None) - 91.0) < 1e-6
    assert abs(best_quali_time("1:32.0", None, None) - 92.0) < 1e-6
    assert best_quali_time(None, None, None) is None


def test_minmax_normalize_range():
    out = minmax_normalize([0.0, 5.0, 10.0], 0.40, 0.95)
    assert abs(out[0] - 0.40) < 1e-9
    assert abs(out[2] - 0.95) < 1e-9
    assert abs(out[1] - 0.675) < 1e-9


def test_minmax_normalize_all_equal():
    out = minmax_normalize([3.0, 3.0], 0.50, 0.95)
    assert out == [0.725, 0.725]


def test_constructor_to_team_aliases():
    assert constructor_to_team("alfa") == "sauber"
    assert constructor_to_team("alphatauri") == "rb"
    assert constructor_to_team("ferrari") == "ferrari"
    assert constructor_to_team("unknown_xyz") == "unknown_xyz"


def test_driver_name_from_id():
    assert driver_name_from_id("max_verstappen") == "Max Verstappen"
    assert driver_name_from_id("leclerc") == "Leclerc"


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
