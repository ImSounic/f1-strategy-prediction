"""Pure tests for eval plan parsing. Run: python tests/test_eval_plans.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.eval.plans import parse_mc_plan, anchor_plan, ACTION_FOR


def test_action_for():
    assert (ACTION_FOR["SOFT"], ACTION_FOR["MEDIUM"], ACTION_FOR["HARD"]) == (1, 2, 3)


def test_parse_mc_two_stop():
    start, plan = parse_mc_plan("2-stop HARD→MEDIUM→SOFT (18/18/21)",
                                "HARD → MEDIUM → SOFT", total_laps=57)
    assert start == "HARD"
    assert plan[0][1] == ACTION_FOR["MEDIUM"] and abs(plan[0][0] - 18 / 57) < 1e-9
    assert plan[1][1] == ACTION_FOR["SOFT"] and abs(plan[1][0] - 36 / 57) < 1e-9
    assert len(plan) == 2


def test_parse_mc_one_stop():
    start, plan = parse_mc_plan("1-stop MEDIUM→HARD (28/29)", "MEDIUM → HARD", total_laps=57)
    assert start == "MEDIUM" and len(plan) == 1 and plan[0][1] == ACTION_FOR["HARD"]


def test_anchor_plans_start_medium_and_switch():
    s1, p1 = anchor_plan("onestop")
    s2, p2 = anchor_plan("twostop")
    assert s1 == "MEDIUM" and p1 == [(0.55, ACTION_FOR["HARD"])]
    assert s2 == "MEDIUM" and len(p2) == 2


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
