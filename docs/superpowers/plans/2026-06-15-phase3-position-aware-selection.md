# Phase 3 — Position-Aware Strategy Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the validated multi-car sim to pick the winner's strategy by **expected finishing position** and contrast it against the **expected-time** pick — measuring whether optimizing for track position predicts the real strategy better than optimizing for raw time.

**Architecture:** A pure helper module (`position_strategy.py`: dedupe candidates by compound sequence, argmin selection) plus a harness (`position_strategy_validation.py`) that, per dry race, runs each deduped candidate for the winner (rest of field on real strategies) and reads `target_time` **and** `target_position` from the *same* sim runs. Time-pick = argmin time; position-pick = argmin position. Each pick is scored against the winner's actual strategy with the validated `strategy_match.score_race`. Aggregated into `results/position_strategy_report.json`.

**Tech Stack:** Python 3.11; pandas + xgboost for the harness (HPC). Pure helper tested on the laptop. Reuses Phase 0/1/2 machinery.

---

## Design (confirmed)

- **Same sim, same candidates, two objectives** — isolates the objective (time vs position) as the only variable.
- **Target = race winner** at real grid; rest of field on real strategies (`greedy_sc=False`).
- **Candidates** = `generate_common_strategies(total_laps)` deduped to one per compound sequence (~14), keeping cost low.
- **Scoring** = `strategy_match.score_race` (stop-count match + full-sequence exact) for each pick vs winner's actual.
- **Caveat (documented):** the "time" objective is the target's *in-race* time (a proxy, includes mild traffic), not clean-air time. Both picks draw from the same candidate set, so both are equally capped by candidate coverage.

## File structure

```
src/analysis/position_strategy.py             # NEW — pure (dedupe + argmin), unit-tested
src/analysis/position_strategy_validation.py  # NEW — harness
tests/test_position_strategy.py               # NEW — pure tests (laptop)
results/position_strategy_report.json         # OUTPUT (HPC)
Makefile                                       # MODIFY — add position-strategy target
```

---

## Task 1: Pure selection helpers

**Files:**
- Create: `src/analysis/position_strategy.py`
- Test: `tests/test_position_strategy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_position_strategy.py`:

```python
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
        _FakeStrategy(["MEDIUM", "HARD"]),   # dup split, same sequence
        _FakeStrategy(["SOFT", "HARD"]),
        _FakeStrategy(["MEDIUM", "HARD", "HARD"]),
    ]
    out = dedupe_by_sequence(cands)
    seqs = [tuple(s.compound_sequence) for s in out]
    assert seqs == [("MEDIUM", "HARD"), ("SOFT", "HARD"), ("MEDIUM", "HARD", "HARD")]


def test_dedupe_normalizes_case_and_skips_empty():
    cands = [
        _FakeStrategy(["medium", "hard"]),
        _FakeStrategy(["MEDIUM", "HARD"]),   # same after normalize -> dropped
        _FakeStrategy([]),                    # empty -> skipped
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_position_strategy.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analysis.position_strategy'`

- [ ] **Step 3: Write the helper module**

Create `src/analysis/position_strategy.py`:

```python
"""
Position-aware strategy selection (pure)
========================================
Helpers for choosing among candidate strategies by a sim-derived objective.
Pure (only depends on the pure strategy_match.normalize_sequence) so it is
unit-testable without the simulator or data.
"""
from __future__ import annotations

from src.analysis.strategy_match import normalize_sequence


def dedupe_by_sequence(strategies):
    """Keep one strategy per distinct compound sequence, preserving order.

    `strategies` is any iterable of objects with a `.compound_sequence`
    attribute (a list like ["MEDIUM", "HARD"] or a "M -> H" string).
    Empty/unknown sequences are skipped.
    """
    seen = set()
    out = []
    for s in strategies:
        key = normalize_sequence(s.compound_sequence)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def argmin_by(items, key):
    """Return the item minimizing item[key], or None if empty."""
    if not items:
        return None
    return min(items, key=lambda d: d[key])
```

- [ ] **Step 4: Run to verify it passes**

Run: `python tests/test_position_strategy.py`
Expected: `4/4 passed`, exit 0

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: Position-aware selection harness

**Files:**
- Create: `src/analysis/position_strategy_validation.py`

- [ ] **Step 1: Write the harness**

Create `src/analysis/position_strategy_validation.py`:

```python
"""
Position-aware strategy selection — validation
==============================================
For each dry race, evaluate candidate strategies for the WINNER (rest of field
on real strategies) and pick by two objectives from the same sim runs:
  - time-optimal     = min expected in-race time
  - position-optimal = min expected finishing position
Each pick is scored against the winner's actual strategy (stop-count + full
compound sequence). Reports whether position-awareness predicts reality better.

Output: results/position_strategy_report.json

Usage:
    python -m src.analysis.position_strategy_validation --seasons 2022 2023 2024 2025 --n-sims 15
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy, generate_common_strategies
from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params
from src.simulation.regulation_profiles import get_profile
from src.analysis.position_validation import load_config, reconstruct_field
from src.analysis.position_strategy import dedupe_by_sequence, argmin_by
from src.analysis.strategy_match import score_race

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def evaluate_candidates(circuit, grid_drivers, strategies, target_idx,
                        candidates, n_sims, base_seed, profile):
    stats = []
    for cand in candidates:
        tsum, psum = 0.0, 0
        for i in range(n_sims):
            sim = MultiCarRaceSim(
                circuit=circuit, drivers=grid_drivers, strategies=strategies,
                target_driver_idx=target_idx, target_strategy=cand,
                greedy_sc=False, profile=profile,
            )
            r = sim.run(seed=base_seed + i)
            tsum += r["target_time"]
            psum += r["target_position"]
        stats.append({
            "seq": list(cand.compound_sequence),
            "num_stops": cand.num_stops,
            "mean_time": tsum / n_sims,
            "mean_pos": psum / n_sims,
        })
    return stats


def _score_pick(pick, real):
    """stop-count + full-sequence match of a pick vs the actual strategy."""
    sim_results = [{"compound_sequence": pick["seq"], "num_stops": pick["num_stops"]}]
    s = score_race(sim_results, real, top_ks=(3,))
    return bool(s["stop_match"]), bool(s["strategy_exact"])


def evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile):
    code_to_driver = {d.code: d for d in drivers}
    field = [f for f in field if f["code"] in code_to_driver]
    winners = [f for f in field if f["finish"] == 1]
    if not winners or len(field) < 2:
        return None
    winner = winners[0]

    field = sorted(field, key=lambda f: f["grid"])
    grid_drivers = [code_to_driver[f["code"]] for f in field]
    strategies = [Strategy(stints=f["stints"], name=f["code"]) for f in field]
    target_idx = next(i for i, f in enumerate(field) if f["code"] == winner["code"])

    candidates = dedupe_by_sequence(generate_common_strategies(circuit.total_laps))
    if not candidates:
        return None

    stats = evaluate_candidates(circuit, grid_drivers, strategies, target_idx,
                                candidates, n_sims, 2000, profile)
    time_pick = argmin_by(stats, "mean_time")
    pos_pick = argmin_by(stats, "mean_pos")

    real = {"compounds": [n for n, _ in winner["stints"]],
            "n_stops": len(winner["stints"]) - 1}
    t_stop, t_exact = _score_pick(time_pick, real)
    p_stop, p_exact = _score_pick(pos_pick, real)

    return {
        "season": season, "round": int(rnd), "winner": winner["code"],
        "actual_seq": real["compounds"], "actual_stops": real["n_stops"],
        "time_pick_seq": time_pick["seq"], "pos_pick_seq": pos_pick["seq"],
        "time_stop_match": t_stop, "time_strat_exact": t_exact,
        "pos_stop_match": p_stop, "pos_strat_exact": p_exact,
    }


def _rates(races):
    n = len(races)
    if n == 0:
        return {}
    def rate(k):
        return round(sum(1 for r in races if r[k]) / n, 3)
    return {
        "n_races": n,
        "time_stop_rate": rate("time_stop_match"),
        "time_strat_rate": rate("time_strat_exact"),
        "pos_stop_rate": rate("pos_stop_match"),
        "pos_strat_rate": rate("pos_strat_exact"),
    }


def run(seasons, n_sims, config_path="configs/config.yaml"):
    config = load_config(config_path)
    raw = config["paths"]["raw"]
    circuit_csv = Path(raw["supplementary"]) / "pirelli_circuit_characteristics.csv"
    circuits_df = pd.read_csv(circuit_csv)
    results_df = pd.read_parquet(Path(raw["jolpica"]) / "results.parquet")
    stints_df = pd.read_parquet(Path("data/features") / "stint_features.parquet")

    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    with open("models/comparison_results.json") as f:
        feature_cols = json.load(f)["experiment"]["feature_columns"]

    all_races = []
    season_reports = []
    for season in seasons:
        profile = get_profile(season)
        drivers, _teams, overtaking = load_drivers(f"configs/drivers_{season}.json")
        rounds = sorted(results_df[results_df["season"] == season]["round"].unique())
        races = []
        for rnd in rounds:
            field = reconstruct_field(season, rnd, stints_df, results_df, circuits_df)
            if field is None:
                continue
            ckey = circuits_df[(circuits_df["season"] == season) &
                               (circuits_df["round_number"] == rnd)].iloc[0]["circuit_key"]
            try:
                circuit = load_circuit_as_params(ckey, season, config, overtaking,
                                                 deg_model, feature_cols)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  {season} r{rnd} ({ckey}): circuit load failed: {e}")
                continue
            row = evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile)
            if row is None:
                continue
            row["circuit"] = ckey
            races.append(row)
            logger.info(f"  {season} r{rnd:>2} {ckey:<14} "
                        f"time[stop={row['time_stop_match']:d} exact={row['time_strat_exact']:d}] "
                        f"pos[stop={row['pos_stop_match']:d} exact={row['pos_strat_exact']:d}]")
        season_reports.append({"season": season, **_rates(races), "races": races})
        all_races.extend(races)
        logger.info(f"=== {season}: {_rates(races)}")

    overall = _rates(all_races)
    report = {
        "methodology": {
            "description": "Per dry race, evaluate deduped candidate strategies for the "
                           "winner (field on real strategies); pick by min expected time vs "
                           "min expected finishing position from the same sim runs; score each "
                           "pick vs the winner's actual strategy.",
            "n_sims": n_sims,
            "caveat": "time objective uses in-race target time (proxy); both picks share the "
                      "same candidate set, equally capped by candidate coverage.",
        },
        "overall": overall,
        "seasons": season_reports,
    }
    out = Path("results/position_strategy_report.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nOVERALL {overall}")
    logger.info(f"Saved: {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Position-aware strategy selection validation")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--n-sims", type=int, default=15)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run(args.seasons, args.n_sims, args.config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `python -m py_compile src/analysis/position_strategy_validation.py src/analysis/position_strategy.py`
Expected: no output, exit 0

- [ ] **Step 3 (HPC): Smoke run**

Run (in `f1-strategy`): `python -m src.analysis.position_strategy_validation --seasons 2024 --n-sims 5`
Expected: per-race `time[...] pos[...]` lines and a season `_rates` summary; `results/position_strategy_report.json` written. Capture any crash.

- [ ] **Step 4 (HPC): Full run**

Run: `make position-strategy`
Inspect the `overall` block: compare `pos_strat_rate` vs `time_strat_rate` (and the stop rates). The headline: does the position objective match the winner's actual strategy at least as often as the time objective?

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 3: Makefile target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add `position-strategy` to `.PHONY` and a target**

Add `position-strategy` to the `.PHONY` list. After the `position-validate` target, add:

```makefile
# ── Position-aware strategy selection (time vs position objective) ────
position-strategy:
	$(PYTHON) src.analysis.position_strategy_validation --seasons 2022 2023 2024 2025 --n-sims 15
	@echo "✓ Position-aware selection report written to results/position_strategy_report.json"
```

- [ ] **Step 2: Commit** (skipped — user commits at end)

---

## Self-review

**Spec coverage (Phase 3):**
- "pick strategy by expected finishing position" → `argmin_by(stats, "mean_pos")`. ✓
- "report alongside time-optimal; measure improvement" → both picks scored + per-season/overall rates. ✓
- Reuse validated `strategy_match` + Phase 2 `reconstruct_field` + era profile → imported. ✓
- Cost control (dedupe candidates, n-sims default 15) → `dedupe_by_sequence`, default. ✓
- Makefile target → Task 3. ✓

**Placeholder scan:** none — full code provided.

**Type consistency:** `dedupe_by_sequence`/`argmin_by` signatures match module/harness/tests. `score_race(sim_results, real, top_ks)` matches `strategy_match` (sim_results entries need `compound_sequence` + `num_stops`; real needs `compounds` + `n_stops`). `MultiCarRaceSim(... profile=...)` matches Phase 0. `generate_common_strategies(total_laps)` and `Strategy(stints=...)` match `multi_car_sim`. `reconstruct_field`/`load_config` imported from `position_validation` (Phase 2).
