"""Pure tests for position scoring. Run: python tests/test_position_match.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.position_match import spearman, score_positions


def test_spearman_perfect():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9


def test_spearman_reversed():
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9


def test_spearman_monotonic_nonlinear():
    assert abs(spearman([1, 2, 3], [10, 20, 90]) - 1.0) < 1e-9


def test_score_positions_perfect():
    pred = {"A": 1, "B": 2, "C": 3}
    act = {"A": 1, "B": 2, "C": 3}
    r = score_positions(pred, act)
    assert abs(r["spearman"] - 1.0) < 1e-9
    assert r["position_mae"] == 0.0
    assert r["n"] == 3


def test_score_positions_one_swap_mae():
    pred = {"A": 1, "B": 2, "C": 3}
    act = {"A": 1, "B": 3, "C": 2}
    r = score_positions(pred, act)
    assert abs(r["position_mae"] - (0 + 1 + 1) / 3) < 1e-9


def test_score_positions_common_keys_only():
    pred = {"A": 1, "B": 2, "C": 3, "D": 4}
    act = {"A": 1, "B": 2, "C": 3}
    r = score_positions(pred, act)
    assert r["n"] == 3


def test_score_positions_too_few():
    r = score_positions({"A": 1}, {"A": 1})
    assert r["spearman"] is None
    assert r["n"] == 1


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
