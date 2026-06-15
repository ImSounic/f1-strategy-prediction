"""
Tests for strategy-matching metrics (src/analysis/strategy_match.py).

Pure-Python; no ML deps required. Run either with pytest or directly:

    python tests/test_strategy_match.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.strategy_match import normalize_sequence, score_race


def _sim(seq, t):
    """Build a fake monte-carlo result dict (only the fields score_race reads)."""
    compounds = seq if isinstance(seq, list) else seq.split("→")
    return {
        "compound_sequence": " → ".join(c.strip() for c in compounds),
        "num_stops": len(compounds) - 1,
        "median_time": t,
    }


def test_normalize_accepts_list_and_string():
    assert normalize_sequence(["SOFT", "HARD"]) == ("SOFT", "HARD")
    assert normalize_sequence("SOFT → HARD") == ("SOFT", "HARD")
    assert normalize_sequence("SOFT->HARD") == ("SOFT", "HARD")
    # equal regardless of representation
    assert normalize_sequence(["MEDIUM", "HARD"]) == normalize_sequence("MEDIUM → HARD")


def test_normalize_drops_unknown_and_blanks():
    assert normalize_sequence("Unknown") == ()
    assert normalize_sequence([]) == ()
    assert normalize_sequence(None) == ()


def test_stop_match_uses_count_only():
    # top pick is 1-stop; real is 1-stop -> stop_match True even if compounds differ
    sims = [_sim("SOFT→HARD", 100.0), _sim("MEDIUM→HARD", 101.0)]
    real = {"n_stops": 1, "compounds": ["MEDIUM", "HARD"]}
    r = score_race(sims, real)
    assert r["stop_match"] is True
    assert r["strategy_exact"] is False     # SOFT→HARD != MEDIUM→HARD


def test_strategy_exact_match():
    sims = [_sim("MEDIUM→HARD", 100.0), _sim("SOFT→HARD", 101.0)]
    real = {"n_stops": 1, "compounds": ["MEDIUM", "HARD"]}
    r = score_race(sims, real)
    assert r["strategy_exact"] is True
    assert r["strategy_top3"] is True
    assert r["strategy_top5"] is True


def test_top5_can_exceed_exact():
    # The bug the old code could NOT express: real strategy is the 4th-best
    # DISTINCT sequence -> not exact, not top3, but IS top5.
    sims = [
        _sim("SOFT→HARD", 100.0),
        _sim("MEDIUM→HARD", 101.0),
        _sim("HARD→MEDIUM", 102.0),
        _sim("SOFT→MEDIUM→HARD", 103.0),   # rank 4 (distinct)
        _sim("MEDIUM→SOFT→HARD", 104.0),
    ]
    real = {"n_stops": 2, "compounds": ["SOFT", "MEDIUM", "HARD"]}
    r = score_race(sims, real)
    assert r["strategy_exact"] is False
    assert r["strategy_top3"] is False
    assert r["strategy_top5"] is True       # <-- impossible to detect with num_stops-only


def test_topk_dedupes_by_sequence():
    # Five lap-split variants of ONE sequence, then a different one.
    # Without dedup, top5 would only cover sequence A and miss B.
    sims = [
        _sim("SOFT→HARD", 100.0),
        _sim("SOFT→HARD", 100.5),
        _sim("SOFT→HARD", 101.0),
        _sim("SOFT→HARD", 101.5),
        _sim("SOFT→HARD", 102.0),
        _sim("MEDIUM→HARD", 102.5),   # 6th raw result, but 2nd DISTINCT sequence
    ]
    real = {"n_stops": 1, "compounds": ["MEDIUM", "HARD"]}
    r = score_race(sims, real)
    assert r["strategy_exact"] is False
    assert r["strategy_top3"] is True     # 2nd distinct sequence is within top-3
    assert r["strategy_top5"] is True


def test_monotonicity_exact_implies_topk():
    sims = [_sim("MEDIUM→HARD", 100.0), _sim("SOFT→HARD", 101.0)]
    real = {"n_stops": 1, "compounds": ["MEDIUM", "HARD"]}
    r = score_race(sims, real)
    assert not (r["strategy_exact"] and not r["strategy_top3"])
    assert not (r["strategy_top3"] and not r["strategy_top5"])


def test_empty_sim_results():
    r = score_race([], {"n_stops": 1, "compounds": ["SOFT", "HARD"]})
    assert r["stop_match"] is False
    assert r["strategy_exact"] is False
    assert r["strategy_top5"] is False


def test_unknown_real_sequence_no_false_positive():
    # Wet/unknown actual compounds -> never a strategy match, but stop_match still works.
    sims = [_sim("SOFT→HARD", 100.0)]
    real = {"n_stops": 1, "compounds": []}
    r = score_race(sims, real)
    assert r["stop_match"] is True
    assert r["strategy_exact"] is False
    assert r["strategy_top5"] is False


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
