"""Pure tests for eval metrics. Run: python tests/test_eval_metrics.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.eval.metrics import win_rate, mean_finish, bootstrap_ci


def test_win_rate_paired():
    assert win_rate([1, 2, 1, 5], [3, 4, 2, 1]) == 0.75


def test_win_rate_ties_excluded():
    assert win_rate([2, 2], [2, 1]) == 0.0


def test_mean_finish():
    assert mean_finish([2, 4, 6]) == 4.0


def test_bootstrap_ci_deterministic_and_brackets_mean():
    lo, hi = bootstrap_ci([2.0, 4.0, 6.0, 8.0], n=500, seed=0)
    lo2, hi2 = bootstrap_ci([2.0, 4.0, 6.0, 8.0], n=500, seed=0)
    assert (lo, hi) == (lo2, hi2)
    assert lo <= 5.0 <= hi


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
