# Phase 1 — Driver-Config Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducibly derive per-season driver ratings (pace_delta, overtaking, tyre_management, team, teammate) from committed data, for all seasons (2022–2025, and 2026 later), replacing the hand-made config files.

**Architecture:** A pure helper module (`driver_config_helpers.py`, unit-tested locally without data) plus a generator (`generate_driver_configs.py`) that reads `qualifying.parquet`, `results.parquet`, `stint_features.parquet`, computes the three ratings, maps constructors→team keys across eras, resolves teammates, and writes `drivers_<season>.json`. The generator writes to `configs/generated/` by default so output can be compared against the committed 2024/25 configs before promotion (acceptance gate / fallback).

**Tech Stack:** Python 3.11, pandas (generator only; needs the `f1-strategy` env on HPC). Helper tests are plain-Python, runnable on the laptop.

---

## Data facts (verified on HPC, 2026-06-15)

- `qualifying.parquet`: `season, round, raceName, circuitId, driverId, driverCode, constructorId, position, Q1, Q2, Q3`. Q-times `"M:SS.mmm"`; one row per driver per round.
- `results.parquet`: `…, driverCode, constructorId, grid, position (None for DNF), …`. ConstructorIds 2022–23: `alfa, alphatauri, …`; 2024–25: `sauber, rb, …`. 15 rows with `grid==0`.
- `stint_features.parquet`: `Season, RoundNumber, Driver, Team, StintNumber, Compound (C1–C5), StintLength, …`.

## Methodology (reverse-engineered from committed 2025 config)

- **pace_delta**: per (round, driver) best quali time via session priority **Q3 > Q2 > Q1**; `gap = best − pole(round)`; per-driver **median gap**; subtract the minimum so the fastest driver = `0.0`. Round 2 dp.
- **overtaking**: per race `positions_gained = grid_eff − finish` (grid 0 → 20; DNF excluded); per-driver **mean**; **min-max normalize to [0.40, 0.95]**. Round 2 dp.
- **tyre_management**: stints with `StintLength ≥ 5`; `ratio = StintLength / median_StintLength_for_that_compound(season)`; per-driver **median ratio**; **min-max normalize to [0.50, 0.95]**. Round 2 dp.
- **team**: most frequent `constructorId` per driver → canonical key (`alfa→sauber`, `alphatauri→rb`, else identity).
- **teammate**: same-team driver with the most shared rounds.
- **name**: best-effort from `driverId` (`max_verstappen → "Max Verstappen"`).

## Acceptance & fallback

Generate to `configs/generated/`. Compare generated 2024/25 vs committed per driver.
- If `pace_delta` within ±0.10s and `overtaking`/`tyre_management` within ±0.10 for most drivers → promote generated (copy over committed) and add 2022/23.
- Else → **keep committed 2024/25**, copy only the new `drivers_2022.json` / `drivers_2023.json` into `configs/`. (Approved fallback.)

## Known limitations (documented in output `notes`)

- Team keys canonicalized to current lineage (`alfa`→Kick Sauber, `alphatauri`→Racing Bulls); display branding uses current names.
- Driver names best-effort from `driverId`.
- `circuit_overtaking_difficulty` is a circuit constant carried across seasons; circuits absent from the table fall back to 0.5 at sim time.

## File structure

```
src/preparation/driver_config_helpers.py   # NEW — pure, unit-tested
src/preparation/generate_driver_configs.py  # NEW — generator (pandas)
tests/test_driver_config_helpers.py         # NEW — pure tests (laptop)
configs/generated/drivers_<season>.json     # OUTPUT (review before promote)
Makefile                                     # MODIFY — add gen-configs target
```

---

## Task 1: Pure helper module

**Files:**
- Create: `src/preparation/driver_config_helpers.py`
- Test: `tests/test_driver_config_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_driver_config_helpers.py`:

```python
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
    # Q3 preferred over Q2 over Q1
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_driver_config_helpers.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.preparation.driver_config_helpers'`

- [ ] **Step 3: Write the helper module**

Create `src/preparation/driver_config_helpers.py`:

```python
"""
Pure helpers for the driver-config generator
============================================
No pandas / no I/O — unit-testable in isolation on any machine.
"""
from __future__ import annotations

# ConstructorId -> canonical team key. Historical lineages are folded onto the
# current key so the simulator's team grouping is consistent across eras.
CONSTRUCTOR_TO_TEAM = {
    "alfa": "sauber",          # Alfa Romeo -> Kick Sauber lineage
    "alphatauri": "rb",        # AlphaTauri -> Racing Bulls lineage
}


def parse_laptime(value) -> float | None:
    """Parse 'M:SS.mmm' or plain seconds into seconds. Blank/None -> None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if ":" in s:
        mins, secs = s.split(":", 1)
        try:
            return int(mins) * 60 + float(secs)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def best_quali_time(q1, q2, q3) -> float | None:
    """Best quali time by session priority Q3 > Q2 > Q1 (first one set)."""
    for q in (q3, q2, q1):
        t = parse_laptime(q)
        if t is not None:
            return t
    return None


def minmax_normalize(values, lo: float, hi: float) -> list:
    """Min-max scale values into [lo, hi]. All-equal -> midpoint."""
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        mid = round((lo + hi) / 2, 3)
        return [mid for _ in values]
    span = vmax - vmin
    return [lo + (v - vmin) * (hi - lo) / span for v in values]


def constructor_to_team(constructor_id: str) -> str:
    """Map a Jolpica constructorId to a canonical team key."""
    return CONSTRUCTOR_TO_TEAM.get(constructor_id, constructor_id)


def driver_name_from_id(driver_id: str) -> str:
    """Best-effort display name from a Jolpica driverId."""
    return driver_id.replace("_", " ").title()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python tests/test_driver_config_helpers.py`
Expected: `7/7 passed`, exit 0

- [ ] **Step 5: Commit** (skipped — user commits manually at end)

---

## Task 2: Generator module

**Files:**
- Create: `src/preparation/generate_driver_configs.py`

- [ ] **Step 1: Write the generator**

Create `src/preparation/generate_driver_configs.py`:

```python
"""
Driver config generator
========================
Reproducibly derives per-season driver ratings from committed Jolpica/feature
data, matching the documented 2024/25 methodology. Writes to configs/generated/
by default so output can be reviewed before replacing committed files.

Usage:
    python -m src.preparation.generate_driver_configs --seasons 2022 2023 2024 2025
    python -m src.preparation.generate_driver_configs --seasons 2026 --out-dir configs
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.preparation.driver_config_helpers import (
    best_quali_time, minmax_normalize, constructor_to_team, driver_name_from_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MIN_STINT_LAPS = 5

# Canonical team metadata + circuit overtaking difficulty (circuit constant),
# sourced from the committed 2025 config. Used for all seasons.
TEAMS = {
    "mclaren": {"color": "#FF8000", "name": "McLaren"},
    "red_bull": {"color": "#3671C6", "name": "Red Bull Racing"},
    "ferrari": {"color": "#E8002D", "name": "Ferrari"},
    "mercedes": {"color": "#27F4D2", "name": "Mercedes"},
    "aston_martin": {"color": "#229971", "name": "Aston Martin"},
    "alpine": {"color": "#FF87BC", "name": "Alpine"},
    "williams": {"color": "#64C4FF", "name": "Williams"},
    "rb": {"color": "#6692FF", "name": "Racing Bulls"},
    "sauber": {"color": "#52E252", "name": "Kick Sauber"},
    "haas": {"color": "#B6BABD", "name": "Haas"},
}

CIRCUIT_OVERTAKING_DIFFICULTY = {
    "bahrain": 0.70, "jeddah": 0.75, "albert_park": 0.55, "suzuka": 0.40,
    "shanghai": 0.65, "miami": 0.60, "imola": 0.35, "monaco": 0.10,
    "montreal": 0.55, "barcelona": 0.35, "spielberg": 0.55, "silverstone": 0.55,
    "hungaroring": 0.30, "spa": 0.65, "zandvoort": 0.25, "monza": 0.80,
    "baku": 0.65, "singapore": 0.30, "cota": 0.60, "mexico": 0.55,
    "interlagos": 0.65, "las_vegas": 0.70, "lusail": 0.55, "yas_marina": 0.55,
}


def _pace_delta(qual: pd.DataFrame) -> dict:
    rows = []
    for (rnd, drv), g in qual.groupby(["round", "driverCode"]):
        r0 = g.iloc[0]
        t = best_quali_time(r0.get("Q1"), r0.get("Q2"), r0.get("Q3"))
        if t is not None:
            rows.append((rnd, drv, t))
    if not rows:
        return {}
    bt = pd.DataFrame(rows, columns=["round", "driver", "t"])
    pole = bt.groupby("round")["t"].min().rename("pole")
    bt = bt.join(pole, on="round")
    bt["gap"] = bt["t"] - bt["pole"]
    med = bt.groupby("driver")["gap"].median()
    med = med - med.min()
    return {d: round(float(v), 2) for d, v in med.items()}


def _overtaking(res: pd.DataFrame) -> dict:
    df = res[res["position"].notna()].copy()
    df["grid_eff"] = df["grid"].where(df["grid"] > 0, 20)
    df["gained"] = df["grid_eff"] - df["position"]
    mean_gain = df.groupby("driverCode")["gained"].mean()
    drivers = list(mean_gain.index)
    vals = minmax_normalize(mean_gain.tolist(), 0.40, 0.95)
    return {d: round(float(v), 2) for d, v in zip(drivers, vals)}


def _tyre_management(stints: pd.DataFrame) -> dict:
    df = stints[stints["StintLength"] >= MIN_STINT_LAPS].copy()
    comp_med = df.groupby("Compound")["StintLength"].median()
    df["ratio"] = df["StintLength"] / df["Compound"].map(comp_med)
    med = df.groupby("Driver")["ratio"].median()
    drivers = list(med.index)
    vals = minmax_normalize(med.tolist(), 0.50, 0.95)
    return {d: round(float(v), 2) for d, v in zip(drivers, vals)}


def _teams_and_mates(res: pd.DataFrame) -> tuple:
    team_of, name_of = {}, {}
    for drv, g in res.groupby("driverCode"):
        team_of[drv] = constructor_to_team(g["constructorId"].mode().iloc[0])
        name_of[drv] = driver_name_from_id(g["driverId"].mode().iloc[0])
    by_team = {}
    for drv, tk in team_of.items():
        by_team.setdefault(tk, []).append(drv)
    rounds_of = {drv: set(g["round"]) for drv, g in res.groupby("driverCode")}
    mates = {}
    for drv, tk in team_of.items():
        best, best_ov = "", -1
        for other in by_team[tk]:
            if other == drv:
                continue
            ov = len(rounds_of[drv] & rounds_of[other])
            if ov > best_ov:
                best_ov, best = ov, other
        mates[drv] = best
    return team_of, mates, name_of


def generate_season(season: int, raw_dir: Path, features_dir: Path) -> dict:
    qual = pd.read_parquet(raw_dir / "qualifying.parquet")
    res = pd.read_parquet(raw_dir / "results.parquet")
    stints = pd.read_parquet(features_dir / "stint_features.parquet")

    qual = qual[qual["season"] == season]
    res = res[res["season"] == season]
    stints = stints[stints["Season"] == season]

    pace = _pace_delta(qual)
    overt = _overtaking(res)
    tyre = _tyre_management(stints)
    team_of, mates, name_of = _teams_and_mates(res)

    pace_fallback = (max(pace.values()) + 0.1) if pace else 0.0
    drivers = []
    for drv in sorted(team_of.keys()):
        drivers.append({
            "code": drv,
            "name": name_of.get(drv, drv),
            "team": team_of[drv],
            "pace_delta": pace.get(drv, round(pace_fallback, 2)),
            "overtaking": overt.get(drv, 0.40),
            "tyre_management": tyre.get(drv, 0.70),
            "teammate": mates.get(drv, ""),
        })
    drivers.sort(key=lambda d: d["pace_delta"])
    reference = drivers[0]["code"] if drivers else ""

    return {
        "season": season,
        "reference_driver": reference,
        "notes": (f"Auto-generated by generate_driver_configs.py from "
                  f"Jolpica/feature data for {season}. pace_delta = median "
                  f"qualifying gap-to-pole (Q3>Q2>Q1), referenced to fastest "
                  f"driver. overtaking = mean positions gained, normalized "
                  f"0.40-0.95. tyre_management = median stint-length ratio vs "
                  f"compound median, normalized 0.50-0.95. Team keys "
                  f"canonicalized (alfa->sauber, alphatauri->rb); names "
                  f"best-effort from driverId."),
        "data_sources": {
            "pace_delta": "Median per-round qualifying gap to pole (Q3>Q2>Q1)",
            "overtaking": "Mean positions gained per race, normalized 0.40-0.95",
            "tyre_management": "Median stint-length ratio vs compound median, normalized 0.50-0.95",
        },
        "teams": TEAMS,
        "drivers": drivers,
        "circuit_overtaking_difficulty": CIRCUIT_OVERTAKING_DIFFICULTY,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate driver config JSONs")
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw/jolpica")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--out-dir", type=str, default="configs/generated")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        cfg = generate_season(season, raw_dir, features_dir)
        out_path = out_dir / f"drivers_{season}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"  wrote {out_path}  ({len(cfg['drivers'])} drivers, ref={cfg['reference_driver']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `python -m py_compile src/preparation/generate_driver_configs.py`
Expected: no output, exit 0

- [ ] **Step 3 (HPC): Generate all seasons to configs/generated/**

Run (in `f1-strategy` env):
`python -m src.preparation.generate_driver_configs --seasons 2022 2023 2024 2025`
Expected: four `configs/generated/drivers_<season>.json` written, ~20 drivers each.

- [ ] **Step 4 (HPC): Acceptance — compare generated 2024/25 vs committed**

Run:
```bash
python - <<'EOF'
import json
def load(p):
    d = json.load(open(p)); return {x["code"]: x for x in d["drivers"]}
for yr in (2024, 2025):
    gen = load(f"configs/generated/drivers_{yr}.json")
    com = load(f"configs/drivers_{yr}.json")
    print(f"=== {yr} ===")
    for f in ("pace_delta", "overtaking", "tyre_management"):
        diffs = [abs(gen[c][f]-com[c][f]) for c in com if c in gen]
        print(f"  {f}: mean|Δ|={sum(diffs)/len(diffs):.3f} max|Δ|={max(diffs):.3f} n={len(diffs)}")
    print("  missing in gen:", [c for c in com if c not in gen])
EOF
```
Decision: if `pace_delta` max|Δ| ≤ ~0.10 and others mean|Δ| ≤ ~0.10 → promote generated. Else keep committed 2024/25, use generated only for 2022/23.

- [ ] **Step 5: Commit** (skipped — user decides placement + commits)

---

## Task 3: Makefile target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the `gen-configs` target and `.PHONY` entry**

In the `.PHONY` line, add `gen-configs`. After the `setup-rl` target block, add:

```makefile
# ── Driver configs (reproducible, all seasons) ───────────────────────
gen-configs:
	$(PYTHON) src.preparation.generate_driver_configs --seasons 2022 2023 2024 2025
	@echo "✓ Driver configs written to configs/generated/ (review before promoting)"
```

- [ ] **Step 2: Commit** (skipped — user commits manually)

---

## Self-review

**Spec coverage (Phase 1):**
- "reproducible, season-general generator" → Task 2. ✓
- documented methodology (pace/overtaking/tyre) → `_pace_delta`/`_overtaking`/`_tyre_management`. ✓
- "reproduces committed 2024/25 within tolerance; document deviation; fallback keep committed" → Task 2 Step 4 + Acceptance/fallback section. ✓
- Makefile target → Task 3. ✓

**Placeholder scan:** none — full code provided; defaults for missing metrics are explicit.

**Type consistency:** helper names (`best_quali_time`, `minmax_normalize`, `constructor_to_team`, `driver_name_from_id`, `parse_laptime`) match between module, generator import, and tests. Generator reads verified column names (`driverCode`, `constructorId`, `grid`, `position`, `Q1/Q2/Q3`, `Season`, `Driver`, `Compound`, `StintLength`).
