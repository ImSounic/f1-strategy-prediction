"""Pure tests for position-aware selection helpers. Run: python tests/test_position_strategy.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.position_strategy import dedupe_by_sequence, argmin_by


class _FakeStrategy:
    """Duck-typed stand-in for multi_car_sim.Strategy."""
    def __init__(self, seq, num_stops=None):
        self.compound_sequence = seq
        self.num_stops = num_stops if num_stops is not None else len(seq) - 1


def test_dedupe_keeps_one_per_sequence_in_order():
    cands = [
        _FakeStrategy(["MEDIUM", "HARD"]),
        _FakeStrategy(["MEDIUM", "HARD"]),
        _FakeStrategy(["SOFT", "HARD"]),
        _FakeStrategy(["MEDIUM", "HARD", "HARD"]),
    ]
    out = dedupe_by_sequence(cands)
    seqs = [tuple(s.compound_sequence) for s in out]
    assert seqs == [("MEDIUM", "HARD"), ("SOFT", "HARD"), ("MEDIUM", "HARD", "HARD")]


def test_dedupe_normalizes_case_and_skips_empty():
    cands = [
        _FakeStrategy(["medium", "hard"]),
        _FakeStrategy(["MEDIUM", "HARD"]),
        _FakeStrategy([]),
    ]
    out = dedupe_by_sequence(cands)
    assert len(out) == 1


def test_argmin_by_picks_minimum():
    stats = [
        {"seq": ["M", "H"], "mean_time": 100.0, "mean_pos": 3.0},
        {"seq": ["S", "H"], "mean_time": 99.0, "mean_pos": 5.0},
        {"seq": ["M", "H", "M"], "mean_time": 101.0, "mean_pos": 2.0},
    ]
    assert argmin_by(stats, "mean_time")["seq"] == ["S", "H"]
    assert argmin_by(stats, "mean_pos")["seq"] == ["M", "H", "M"]


def test_argmin_by_empty():
    assert argmin_by([], "mean_time") is None


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
