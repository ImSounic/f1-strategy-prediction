# Phase 2 — Position-Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how well `MultiCarRaceSim` predicts real finishing order — feed each race's real grid + every car's real strategy, simulate, and score predicted vs actual order with Spearman rank correlation + position MAE across 2022–2025.

**Architecture:** A pure, unit-tested scorer (`position_match.py`) plus a harness (`position_validation.py`) that reconstructs each race's classified-finisher field (grid from `results.grid`, strategies from `stint_features`), reuses `load_drivers` + `load_circuit_as_params`, runs the sim with all cars on fixed real strategies (`greedy_sc=False`) over N seeds, and aggregates per season + overall into `results/position_validation_report.json`.

**Tech Stack:** Python 3.11; pandas + xgboost for the harness (HPC `f1-strategy` env). `position_match.py` is pure (no deps) and tested on the laptop.

---

## Design decisions (within the approved spec)

- **Finishers only:** the field is built from classified finishers (`results.position` not null). DNF cars are excluded from both the field and the metric (the sim has no retirement model). Documented limitation.
- **Dry races only:** the sim has no wet model. A race is skipped if stint coverage is poor — `sum(winner stint laps) < 0.85 * total_laps` (indicates wet laps filtered out of `stint_features`). Documented.
- **Grid:** `results.grid` (real starting slot incl. penalties); `grid==0` (pit lane) → back of grid.
- **Compound mapping:** `stint_features.Compound` (C-codes) → SOFT/MEDIUM/HARD via the circuit's `soft/medium/hard_compound` columns.
- **Prediction:** mean finishing position per driver across N seeds → rank → predicted order.

## File structure

```
src/analysis/position_match.py            # NEW — pure scorer (spearman + MAE), unit-tested
src/analysis/position_validation.py       # NEW — harness (pandas + sim)
tests/test_position_match.py              # NEW — pure tests (laptop)
results/position_validation_report.json   # OUTPUT (HPC)
Makefile                                   # MODIFY — add position-validate target
```

---

## Task 1: Pure position scorer

**Files:**
- Create: `src/analysis/position_match.py`
- Test: `tests/test_position_match.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_position_match.py`:

```python
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
    # rank correlation ignores scale -> still 1.0
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
    act = {"A": 1, "B": 3, "C": 2}     # B and C off by 1 each
    r = score_positions(pred, act)
    assert abs(r["position_mae"] - (0 + 1 + 1) / 3) < 1e-9


def test_score_positions_common_keys_only():
    pred = {"A": 1, "B": 2, "C": 3, "D": 4}
    act = {"A": 1, "B": 2, "C": 3}      # D missing in actual
    r = score_positions(pred, act)
    assert r["n"] == 3


def test_score_positions_too_few():
    r = score_positions({"A": 1}, {"A": 1})
    assert r["spearman"] is None       # need >=2 to correlate
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_position_match.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analysis.position_match'`

- [ ] **Step 3: Write the scorer**

Create `src/analysis/position_match.py`:

```python
"""
Position-accuracy scoring (pure)
================================
Spearman rank correlation + mean absolute position error, over a common set of
drivers. No numpy/scipy so it is unit-testable anywhere.
"""
from __future__ import annotations


def _mean(xs):
    return sum(xs) / len(xs)


def _ranks(values):
    """Fractional ranks (1-indexed), averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def spearman(x, y):
    """Spearman rank correlation between two equal-length sequences."""
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_ranks(x), _ranks(y))


def score_positions(predicted: dict, actual: dict) -> dict:
    """Score predicted vs actual finishing positions over common drivers.

    predicted/actual: {driver_code: position (1-indexed)}.
    Returns {spearman, position_mae, n}.
    """
    common = sorted(set(predicted) & set(actual))
    n = len(common)
    if n == 0:
        return {"spearman": None, "position_mae": None, "n": 0}
    pred = [predicted[c] for c in common]
    act = [actual[c] for c in common]
    mae = _mean([abs(p - a) for p, a in zip(pred, act)])
    return {"spearman": spearman(pred, act), "position_mae": mae, "n": n}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python tests/test_position_match.py`
Expected: `7/7 passed`, exit 0

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: Position-validation harness

**Files:**
- Create: `src/analysis/position_validation.py`

- [ ] **Step 1: Write the harness**

Create `src/analysis/position_validation.py`:

```python
"""
Position validation harness
===========================
Feeds each race's real grid + every car's real strategy into MultiCarRaceSim
and scores predicted vs actual finishing order (Spearman + position MAE), over
classified finishers, for dry races across the requested seasons.

Output: results/position_validation_report.json

Usage:
    python -m src.analysis.position_validation --seasons 2022 2023 2024 2025 --n-sims 30
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy, build_grid, find_target_in_grid
from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params
from src.simulation.regulation_profiles import get_profile
from src.analysis.position_match import score_positions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DRY_COVERAGE_MIN = 0.85


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def reconstruct_field(season, rnd, stints_df, results_df, circuits_df):
    """Return classified-finisher field for a race, or None if wet/insufficient.

    Each entry: {code, grid, finish, stints=[(name, laps), ...]}.
    """
    crow = circuits_df[(circuits_df["season"] == season) &
                       (circuits_df["round_number"] == rnd)]
    if crow.empty:
        return None
    crow = crow.iloc[0]
    total_laps = int(crow["total_laps"])
    cmap = {crow["soft_compound"]: "SOFT",
            crow["medium_compound"]: "MEDIUM",
            crow["hard_compound"]: "HARD"}

    res = results_df[(results_df["season"] == season) & (results_df["round"] == rnd)]
    res = res[res["position"].notna()]
    if res.empty:
        return None

    field = []
    winner_cover = 0.0
    for _, r in res.iterrows():
        code = r["driverCode"]
        ds = stints_df[(stints_df["Season"] == season) &
                       (stints_df["RoundNumber"] == rnd) &
                       (stints_df["Driver"] == code)].sort_values("StintNumber")
        if ds.empty:
            continue
        stints = [(cmap.get(s["Compound"], "MEDIUM"), int(s["StintLength"]))
                  for _, s in ds.iterrows()]
        cover = sum(n for _, n in stints) / max(total_laps, 1)
        if int(r["position"]) == 1:
            winner_cover = cover
        grid = int(r["grid"]) if r["grid"] > 0 else 20
        field.append({"code": code, "grid": grid, "finish": int(r["position"]), "stints": stints})

    if len(field) < 2:
        return None
    if winner_cover and winner_cover < DRY_COVERAGE_MIN:
        return None  # likely wet (stint laps don't cover the race)
    return field


def simulate_order(circuit, drivers, field, n_sims, base_seed, profile):
    """Run the field on fixed real strategies; return {code: mean finish position}."""
    code_to_driver = {d.code: d for d in drivers}
    field = [f for f in field if f["code"] in code_to_driver]
    if len(field) < 2:
        return None

    # Order grid entries by real grid slot; assign sequential grid positions 1..N
    field = sorted(field, key=lambda f: f["grid"])
    grid_drivers = [code_to_driver[f["code"]] for f in field]
    strategies = [Strategy(stints=f["stints"], name=f["code"]) for f in field]

    sums = {f["code"]: 0 for f in field}
    for i in range(n_sims):
        sim = MultiCarRaceSim(
            circuit=circuit,
            drivers=grid_drivers,
            strategies=strategies,
            target_driver_idx=0,
            target_strategy=strategies[0],
            greedy_sc=False,          # everyone follows their fixed real strategy
            profile=profile,
        )
        result = sim.run(seed=base_seed + i)
        fin = result["finishing_positions"]   # 1-indexed, per grid index
        for idx, f in enumerate(field):
            sums[f["code"]] += fin[idx]

    mean_pos = {c: s / n_sims for c, s in sums.items()}
    # Convert mean positions to integer predicted ranks 1..N
    ordered = sorted(mean_pos, key=lambda c: mean_pos[c])
    return {c: rank + 1 for rank, c in enumerate(ordered)}


def run_validation(seasons, n_sims, config_path="configs/config.yaml"):
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
            predicted = simulate_order(circuit, drivers, field, n_sims, 1000, profile)
            if predicted is None:
                continue
            actual = {f["code"]: f["finish"] for f in field if f["code"] in predicted}
            score = score_positions(predicted, actual)
            score.update({"season": season, "round": int(rnd), "circuit": ckey})
            races.append(score)
            logger.info(f"  {season} r{rnd:>2} {ckey:<14} "
                        f"spearman={score['spearman']:.3f} MAE={score['position_mae']:.2f} n={score['n']}")

        valid = [r for r in races if r["spearman"] is not None]
        mean_sp = float(np.mean([r["spearman"] for r in valid])) if valid else None
        pooled_mae = (float(np.average([r["position_mae"] for r in valid],
                                       weights=[r["n"] for r in valid])) if valid else None)
        season_reports.append({
            "season": season, "n_races": len(valid),
            "mean_spearman": mean_sp, "pooled_position_mae": pooled_mae,
            "races": races,
        })
        logger.info(f"=== {season}: races={len(valid)} mean_spearman={mean_sp} "
                    f"pooled_MAE={pooled_mae}")

    all_valid = [r for s in season_reports for r in s["races"] if r["spearman"] is not None]
    overall = {
        "n_races": len(all_valid),
        "mean_spearman": float(np.mean([r["spearman"] for r in all_valid])) if all_valid else None,
        "pooled_position_mae": (float(np.average([r["position_mae"] for r in all_valid],
                                                 weights=[r["n"] for r in all_valid]))
                                if all_valid else None),
    }
    report = {
        "methodology": {
            "description": "Real grid + real per-car strategies fed to MultiCarRaceSim; "
                           "predicted vs actual finishing order over classified finishers, dry races.",
            "n_sims": n_sims, "dry_coverage_min": DRY_COVERAGE_MIN,
            "limitations": ["no DNF/retirement model (finishers only)",
                            "wet races skipped via stint coverage",
                            "grid from results.grid; season-aggregate driver form"],
        },
        "overall": overall,
        "seasons": season_reports,
    }
    out = Path("results/position_validation_report.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nOverall: races={overall['n_races']} "
                f"mean_spearman={overall['mean_spearman']} "
                f"pooled_MAE={overall['pooled_position_mae']}")
    logger.info(f"Saved: {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Position validation harness")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--n-sims", type=int, default=30)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_validation(args.seasons, args.n_sims, args.config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `python -m py_compile src/analysis/position_validation.py src/analysis/position_match.py`
Expected: no output, exit 0

- [ ] **Step 3 (HPC): Smoke run on one season, few sims**

Run (in `f1-strategy`): `python -m src.analysis.position_validation --seasons 2024 --n-sims 5`
Expected: per-race lines with `spearman=`/`MAE=`, a season summary, and `results/position_validation_report.json` written. If a KeyError/None appears, capture it for a fix.

- [ ] **Step 4 (HPC): Full run**

Run: `make position-validate` (all seasons, default n-sims).
Inspect `results/position_validation_report.json` `overall` block — sanity: `mean_spearman` should be clearly positive (e.g. > 0.3) since real grid + real strategies should track real order; `pooled_position_mae` a few positions.

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 3: Makefile target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add `position-validate` to `.PHONY` and a target**

Add `position-validate` to the `.PHONY` list. After the `gen-configs` target, add:

```makefile
# ── Position validation (multi-car sim vs real finishing order) ──────
position-validate:
	$(PYTHON) src.analysis.position_validation --seasons 2022 2023 2024 2025 --n-sims 30
	@echo "✓ Position validation written to results/position_validation_report.json"
```

- [ ] **Step 2: Commit** (skipped — user commits at end)

---

## Self-review

**Spec coverage (Phase 2):**
- Full-field reconstruction (Approach A) → `reconstruct_field`. ✓
- Reuse `load_drivers`/`load_circuit_as_params` → imported and used. ✓
- All cars on fixed real strategies, era-aware → `simulate_order` (`greedy_sc=False`, `profile=get_profile(season)`). ✓
- Spearman + position MAE over classified finishers → `position_match.score_positions`, finishers-only field. ✓
- Per-season + overall report → `run_validation` → `results/position_validation_report.json`. ✓
- Limitations documented in report. ✓
- Makefile target → Task 3. ✓

**Placeholder scan:** none — full code provided; wet/DNF handling explicit.

**Type consistency:** `score_positions`/`spearman` signatures match between module, harness, and tests. Harness uses verified columns (`results`: season/round/driverCode/grid/position; `stint_features`: Season/RoundNumber/Driver/StintNumber/Compound/StintLength; circuits: season/round_number/circuit_key/total_laps/soft|medium|hard_compound). `MultiCarRaceSim(profile=...)` matches the Phase 0 signature; `Strategy(stints=[(name, laps)])` matches `multi_car_sim.Strategy`.
