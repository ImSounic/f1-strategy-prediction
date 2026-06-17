"""Pure tests for the RL-2b league core. Run: python tests/test_league.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.league import (
    LeagueConfig, learner_ids, role_of, WinRateMatrix, pfsp_weights,
    opponent_pool, focus_count, assemble_field, league_winrate,
    should_snapshot, should_reset_exploiter,
    ROLE_MAIN, ROLE_MEXP, ROLE_LEXP,
)


# ── Task 1: roles, config, matrix ───────────────────────────────────────────
def test_learner_ids_and_role_of():
    ids = learner_ids(LeagueConfig(n_main=2, n_mexp=2, n_lexp=2))
    assert ids == ["main_0", "main_1", "mexp_0", "mexp_1", "lexp_0", "lexp_1"]
    assert role_of("main_0") == "main"
    assert role_of("snap_3") == "snap"
    assert role_of("anchor_onestop") == "anchor"


def test_winrate_records_pairwise():
    m = WinRateMatrix()
    m.record_race([("main_0", 1), ("anchor_x", 2), ("snap_0", 3)])
    assert m.rate("main_0", "anchor_x") == 1.0
    assert m.rate("anchor_x", "main_0") == 0.0
    assert m.games("main_0", "snap_0") == 1


def test_winrate_prior_when_unseen():
    m = WinRateMatrix()
    assert m.rate("a", "b") == 0.5
    assert m.rate("a", "b", prior=0.3) == 0.3


# ── Task 2: PFSP ─────────────────────────────────────────────────────────────
def test_pfsp_weights_sum_to_one_and_favor_losses():
    w = pfsp_weights([0.1, 0.9], p=2.0, eps=0.0)
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[1]


def test_pfsp_eps_gives_coverage():
    w = pfsp_weights([0.0, 1.0], p=2.0, eps=0.2)
    assert all(x > 0 for x in w)


def test_pfsp_uniform_when_equal():
    w = pfsp_weights([0.5, 0.5, 0.5], p=2.0, eps=0.1)
    assert all(abs(x - 1 / 3) < 1e-9 for x in w)


# ── Task 3: role pool + field ────────────────────────────────────────────────
def test_opponent_pool_by_role():
    learners = ["main_0", "main_1", "mexp_0", "lexp_0"]
    snaps, anchors = ["snap_0"], ["anchor_onestop"]
    assert set(opponent_pool(ROLE_MEXP, "mexp_0", learners, snaps, anchors)) == {"main_0", "main_1"}
    assert set(opponent_pool(ROLE_MAIN, "main_0", learners, snaps, anchors)) == {"snap_0", "anchor_onestop", "main_1"}
    assert "lexp_0" not in opponent_pool(ROLE_LEXP, "lexp_0", learners, snaps, anchors)


def test_focus_count_and_assemble():
    assert focus_count(20, 0.5) == 10
    assert focus_count(1, 0.5) == 1
    field = assemble_field(5, "main_0", n_focus=2, sample_opponent=lambda: "anchor_onestop")
    assert field == ["main_0", "main_0", "anchor_onestop", "anchor_onestop", "anchor_onestop"]


# ── Task 4: snapshot / reset predicates ──────────────────────────────────────
def test_league_winrate_average():
    m = WinRateMatrix()
    m.record_race([("main_0", 1), ("snap_0", 2)])
    m.record_race([("main_0", 2), ("anchor_x", 1)])
    assert abs(league_winrate(m, "main_0", ["snap_0", "anchor_x"]) - 0.5) < 1e-9


def test_snapshot_and_reset_triggers():
    m = WinRateMatrix()
    for _ in range(10):
        m.record_race([("main_0", 1), ("snap_0", 2), ("anchor_x", 3)])
    assert should_snapshot(m, "main_0", ["snap_0", "anchor_x"], steps_since=0,
                           threshold=0.7, every_steps=10**9) is True
    assert should_reset_exploiter(m, "mexp_0", ["main_0"], steps_alive=0,
                                  threshold=0.7, max_steps=10**9) is False
    assert should_reset_exploiter(m, "mexp_0", ["main_0"], steps_alive=10**10,
                                  threshold=0.7, max_steps=10**9) is True


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
