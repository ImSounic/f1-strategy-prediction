# Phase 0 — Regulation-Era Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `RegulationProfile` abstraction so the multi-car simulator's era-specific physics constants are parameterised by season instead of hardcoded, without changing 2022–25 behaviour.

**Architecture:** A new pure module `regulation_profiles.py` defines a frozen `RegulationProfile` dataclass and two era instances (`ground_effect_2022_25`, `new_era_2026`) plus `get_era(season)` / `get_profile(season)`. `MultiCarRaceSim` gains a `profile` parameter (defaulting to the 2022–25 profile) and reads all era constants from it. The 2022–25 profile holds the simulator's *current exact* constants, so existing behaviour is preserved.

**Tech Stack:** Python 3.11, dataclasses, numpy (sim only). Tests are plain-Python (no pytest dependency), mirroring `tests/test_strategy_match.py`.

---

## Scope notes

- **In scope:** `regulation_profiles.py` + refactor of `multi_car_sim.py` (the simulator Phase 2 validates and that 2026 most changes).
- **Deferred to Phase 4 (intentional):** `strategy_simulator.py` keeps its existing **config.yaml-driven** fuel model (already parameterised), and `generate_strategies` keeps using the per-circuit compound names. The era `compound_set` (C1–C5 for 2026) feeds the *degradation model / circuit mapping*, which only diverges in Phase 4 — wiring it earlier would add risk with no 2022–25 benefit.
- **Behaviour preservation is the acceptance bar:** default (no `profile` arg) must produce byte-identical sim results to today.

## File structure

```
src/simulation/regulation_profiles.py   # NEW — pure, no I/O, no ML deps
src/simulation/multi_car_sim.py          # MODIFY — read constants from profile
tests/test_regulation_profiles.py        # NEW — pure unit tests
tests/test_multi_car_sim_profile.py      # NEW — behaviour-preservation tests (numpy only)
```

Both test files are runnable on the laptop (numpy is installed) and on HPC via
`python tests/<file>.py` — no pytest required.

---

## Task 1: Regulation profiles module

**Files:**
- Create: `src/simulation/regulation_profiles.py`
- Test: `tests/test_regulation_profiles.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_regulation_profiles.py`:

```python
"""Tests for regulation-era profiles. Pure Python; run: python tests/test_regulation_profiles.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.regulation_profiles import (
    get_era, get_profile, GROUND_EFFECT_2022_25, NEW_ERA_2026, DEFAULT_PROFILE,
)


def test_get_era_boundaries():
    assert get_era(2022) == "ground_effect_2022_25"
    assert get_era(2025) == "ground_effect_2022_25"
    assert get_era(2026) == "new_era_2026"
    assert get_era(2027) == "new_era_2026"


def test_get_profile_returns_matching_era():
    assert get_profile(2024) is GROUND_EFFECT_2022_25
    assert get_profile(2026) is NEW_ERA_2026


def test_default_profile_is_2022_25():
    assert DEFAULT_PROFILE is GROUND_EFFECT_2022_25


def test_2022_25_profile_pins_legacy_constants():
    p = GROUND_EFFECT_2022_25
    assert p.base_pace == 90.0
    assert p.start_fuel_kg == 110.0
    assert p.fuel_effect_per_kg == 0.035
    assert p.sc_pace_factor == 1.40
    assert p.vsc_pace_factor == 1.20
    assert p.compound_deg_base == {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04}
    assert p.compound_cliff == {"SOFT": 20, "MEDIUM": 30, "HARD": 40}
    assert p.dirty_air_window == 1.5
    assert p.dirty_air_penalty == 0.15
    assert p.drs_window == 1.0
    assert p.overtake_aid_benefit == 0.3
    assert p.lap_time_noise_std == 0.3
    assert p.compound_set == ("C1", "C2", "C3", "C4", "C5", "C6")
    assert p.overtaking_mode == "drs"


def test_2026_profile_drops_c6_and_changes_overtaking():
    p = NEW_ERA_2026
    assert "C6" not in p.compound_set
    assert p.compound_set == ("C1", "C2", "C3", "C4", "C5")
    assert p.overtaking_mode == "override_boost"
    assert p.dirty_air_penalty < GROUND_EFFECT_2022_25.dirty_air_penalty


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_regulation_profiles.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.regulation_profiles'`

- [ ] **Step 3: Write the module**

Create `src/simulation/regulation_profiles.py`:

```python
"""
Regulation-era profiles
=======================
F1 has periodic regulation resets that change the physics the simulator assumes
(tyre construction, fuel/energy, aerodynamics/overtaking). Reusing one era's
constants across a reset biases predictions. A ``RegulationProfile`` bundles the
era-specific constants so the simulator is parameterised by season instead of
hardcoding one era.

Eras:
  ground_effect_2022_25 — ground-effect cars, DRS, C1-C6 tyres (2022-2025).
  new_era_2026          — 2026 reset: active aero + override boost (no DRS),
                          C1-C5 tyres, ~50/50 hybrid PU. The physics constants
                          here are SEEDED from the 2022-25 baseline and are
                          CALIBRATED against real 2026 data in Phase 4.

The 2022-25 profile reproduces the multi-car simulator's original constants
exactly, so existing behaviour (and validation on 2022-25) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegulationProfile:
    """Era-specific physics constants for the multi-car simulator."""
    name: str
    seasons: tuple              # seasons this era covers
    compound_set: tuple         # dry compound codes available (C1..)
    base_pace: float            # reference lap time (s)
    start_fuel_kg: float
    fuel_effect_per_kg: float   # s/lap per kg of fuel on board
    sc_pace_factor: float       # lap-time multiplier under full SC
    vsc_pace_factor: float      # lap-time multiplier under VSC
    compound_deg_base: dict     # {compound_name: base deg rate s/lap}
    compound_cliff: dict        # {compound_name: cliff lap}
    dirty_air_window: float     # gap (s) within which dirty air bites
    dirty_air_penalty: float    # max s/lap lost in dirty air
    drs_window: float           # gap (s) within which DRS/override applies
    overtake_aid_benefit: float # s/lap benefit from DRS/override at full effect
    lap_time_noise_std: float   # per-lap gaussian noise std (s)
    overtaking_mode: str        # "drs" | "override_boost"


GROUND_EFFECT_2022_25 = RegulationProfile(
    name="ground_effect_2022_25",
    seasons=(2022, 2023, 2024, 2025),
    compound_set=("C1", "C2", "C3", "C4", "C5", "C6"),
    base_pace=90.0,
    start_fuel_kg=110.0,
    fuel_effect_per_kg=0.035,
    sc_pace_factor=1.40,
    vsc_pace_factor=1.20,
    compound_deg_base={"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04},
    compound_cliff={"SOFT": 20, "MEDIUM": 30, "HARD": 40},
    dirty_air_window=1.5,
    dirty_air_penalty=0.15,
    drs_window=1.0,
    overtake_aid_benefit=0.3,
    lap_time_noise_std=0.3,
    overtaking_mode="drs",
)

# 2026 reset. Structural changes are known (C6 dropped; override boost replaces
# DRS; closer following from -55% drag). Numeric physics constants are seeded
# from 2022-25 and CALIBRATED against real 2026 data in Phase 4 — only the
# dirty-air penalty is pre-reduced as a placeholder to reflect closer following.
NEW_ERA_2026 = RegulationProfile(
    name="new_era_2026",
    seasons=(2026,),
    compound_set=("C1", "C2", "C3", "C4", "C5"),
    base_pace=90.0,
    start_fuel_kg=110.0,
    fuel_effect_per_kg=0.035,
    sc_pace_factor=1.40,
    vsc_pace_factor=1.20,
    compound_deg_base={"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04},
    compound_cliff={"SOFT": 20, "MEDIUM": 30, "HARD": 40},
    dirty_air_window=1.0,
    dirty_air_penalty=0.07,
    drs_window=1.0,
    overtake_aid_benefit=0.3,
    lap_time_noise_std=0.3,
    overtaking_mode="override_boost",
)

ERA_PROFILES = (GROUND_EFFECT_2022_25, NEW_ERA_2026)
DEFAULT_PROFILE = GROUND_EFFECT_2022_25


def get_era(season: int) -> str:
    """Return the regulation-era name for a season."""
    if season >= 2026:
        return "new_era_2026"
    return "ground_effect_2022_25"


def get_profile(season: int) -> RegulationProfile:
    """Return the RegulationProfile for a season."""
    name = get_era(season)
    for profile in ERA_PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"No regulation profile for season {season}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_regulation_profiles.py`
Expected: `6/6 passed`, exit 0

- [ ] **Step 5: Commit**

```bash
git add src/simulation/regulation_profiles.py tests/test_regulation_profiles.py
git commit -m "feat(sim): add regulation-era profile abstraction (Phase 0)"
```

---

## Task 2: Refactor multi_car_sim to read from a profile

**Files:**
- Modify: `src/simulation/multi_car_sim.py`
- Test: `tests/test_multi_car_sim_profile.py`

- [ ] **Step 1: Write the behaviour-preservation test**

Create `tests/test_multi_car_sim_profile.py`:

```python
"""Behaviour-preservation tests for the profile refactor of multi_car_sim.
Needs numpy only. Run: python tests/test_multi_car_sim_profile.py"""
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.multi_car_sim import (
    MultiCarRaceSim, DriverConfig, CircuitParams, Strategy,
)
from src.simulation.regulation_profiles import GROUND_EFFECT_2022_25


def _scenario():
    drivers = [
        DriverConfig("AAA", "A", "t1", 0.0, 0.6, 0.7, "BBB"),
        DriverConfig("BBB", "B", "t1", 0.2, 0.5, 0.6, "AAA"),
        DriverConfig("CCC", "C", "t2", 0.4, 0.5, 0.6, "DDD"),
    ]
    circuit = CircuitParams(
        circuit_key="test", circuit_name="Test", total_laps=20,
        pit_loss_seconds=20.0, sc_prob_per_race=0.0, vsc_prob_per_race=0.0,
        overtaking_difficulty=0.5, deg_rates={},
    )
    strat = Strategy(stints=[("MEDIUM", 10), ("HARD", 10)], name="MH")
    return circuit, drivers, [strat, strat, strat], strat


def test_default_profile_matches_explicit_2022_25():
    circuit, drivers, strategies, strat = _scenario()
    a = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=42)
    b = MultiCarRaceSim(circuit, drivers, strategies, 0, strat,
                        profile=GROUND_EFFECT_2022_25).run(seed=42)
    assert a["target_time"] == b["target_time"]
    assert a["finishing_positions"] == b["finishing_positions"]


def test_deterministic_same_seed():
    circuit, drivers, strategies, strat = _scenario()
    a = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=7)
    b = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=7)
    assert a["target_time"] == b["target_time"]
    assert a["finishing_positions"] == b["finishing_positions"]


def test_sim_reads_profile_constants():
    # Lowering base_pace in the profile MUST lower simulated time -> proves the
    # sim reads the profile rather than stale module-level constants.
    circuit, drivers, strategies, strat = _scenario()
    base = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=42)
    faster = replace(GROUND_EFFECT_2022_25, base_pace=80.0)
    out = MultiCarRaceSim(circuit, drivers, strategies, 0, strat,
                          profile=faster).run(seed=42)
    assert out["target_time"] < base["target_time"]


def test_finishing_positions_are_a_permutation():
    circuit, drivers, strategies, strat = _scenario()
    r = MultiCarRaceSim(circuit, drivers, strategies, 0, strat).run(seed=1)
    assert sorted(r["finishing_positions"]) == [1, 2, 3]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_multi_car_sim_profile.py`
Expected: FAIL — `MultiCarRaceSim.__init__() got an unexpected keyword argument 'profile'`

- [ ] **Step 3: Replace the module-level constants block**

In `src/simulation/multi_car_sim.py`, replace the constants block (currently lines ~28–40):

```python
# ── Constants ──────────────────────────────────────────────────

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_DEG_BASE = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04}
COMPOUND_CLIFF = {"SOFT": 20, "MEDIUM": 30, "HARD": 40}

# Base pace reference (same as RL env)
BASE_PACE = 90.0
START_FUEL_KG = 110.0
FUEL_EFFECT_PER_KG = 0.035
SC_PACE_FACTOR = 1.40
VSC_PACE_FACTOR = 1.20
PIT_STOP_STATIONARY = 2.5  # seconds stationary (added on top of pit_loss)
```

with:

```python
from src.simulation.regulation_profiles import RegulationProfile, DEFAULT_PROFILE

# ── Constants ──────────────────────────────────────────────────
# Era-specific physics now live in RegulationProfile (see regulation_profiles.py).
# These module-level names are kept for backward compatibility (e.g.
# precompute_scenarios imports COMPOUND_DEG_BASE) and derive from the default
# (2022-25) profile so there is a single source of truth.
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]            # compound NAMES (era-independent)
COMPOUND_DEG_BASE = DEFAULT_PROFILE.compound_deg_base
COMPOUND_CLIFF = DEFAULT_PROFILE.compound_cliff
```

- [ ] **Step 4: Add the `profile` parameter to `MultiCarRaceSim.__init__`**

In `__init__`, change the signature to add `profile` after `greedy_sc`:

```python
    def __init__(
        self,
        circuit: CircuitParams,
        drivers: list[DriverConfig],
        strategies: list[Strategy],  # one per driver, same order
        target_driver_idx: int,
        target_strategy: Strategy,
        greedy_sc: bool = True,
        profile: RegulationProfile = None,
    ):
        self.circuit = circuit
        self.drivers = drivers
        self.n_cars = len(drivers)
        self.strategies = strategies
        self.target_idx = target_driver_idx
        self.target_strategy = target_strategy
        self.greedy_sc = greedy_sc
        self.profile = profile if profile is not None else DEFAULT_PROFILE
```

Then change the fuel burn-rate line in `__init__` from:

```python
        self.burn_rate = START_FUEL_KG / tl
```
to:
```python
        self.burn_rate = self.profile.start_fuel_kg / tl
```

- [ ] **Step 5: Route `_get_deg_rate` through the profile**

Change:
```python
        base = circuit.deg_rates.get(compound, COMPOUND_DEG_BASE[compound])
```
to:
```python
        base = circuit.deg_rates.get(compound, self.profile.compound_deg_base[compound])
```

- [ ] **Step 6: Route `_compute_lap_time` through the profile**

Replace the body of `_compute_lap_time` (from the SC check to the return) with:

```python
        if sc_active:
            return self.profile.base_pace * self.profile.sc_pace_factor
        if vsc_active:
            return self.profile.base_pace * self.profile.vsc_pace_factor

        # Base + driver delta
        lap_time = self.profile.base_pace + driver.pace_delta

        # Fuel effect
        fuel_remaining = max(0, self.profile.start_fuel_kg - self.burn_rate * (lap - 1))
        lap_time += fuel_remaining * self.profile.fuel_effect_per_kg

        # Tyre degradation (quadratic model matching RL env)
        deg_rate = self._get_deg_rate(car.tyre_compound, driver, self.circuit)
        tyre_deg = deg_rate * car.tyre_age + 0.002 * (car.tyre_age ** 1.3)
        lap_time += tyre_deg

        # Dirty air penalty (within the era's dirty-air window of car ahead)
        if dirty_air and gap_to_ahead < self.profile.dirty_air_window:
            lap_time += self.profile.dirty_air_penalty * (
                self.profile.dirty_air_window - gap_to_ahead
            ) / self.profile.dirty_air_window

        # Overtaking aid (DRS / 2026 override) within the era's window of car ahead
        if 0 < gap_to_ahead < self.profile.drs_window:
            drs_benefit = self.profile.overtake_aid_benefit * self.circuit.overtaking_difficulty
            lap_time -= drs_benefit

        # Random variation
        lap_time += rng.normal(0, self.profile.lap_time_noise_std)

        return max(lap_time, self.profile.base_pace * 0.95)  # floor
```

- [ ] **Step 7: Route `_should_pit_greedy_sc` cliff through the profile**

Change:
```python
        cliff = COMPOUND_CLIFF.get(car.tyre_compound, 30)
```
to:
```python
        cliff = self.profile.compound_cliff.get(car.tyre_compound, 30)
```

- [ ] **Step 8: Route the `run()` dirty-air flag through the profile**

In `run()`, change:
```python
                dirty_air = gaps[i] < 1.5 and not sc_active and not vsc_active
```
to:
```python
                dirty_air = (gaps[i] < self.profile.dirty_air_window
                             and not sc_active and not vsc_active)
```

- [ ] **Step 9: Run the behaviour-preservation tests**

Run: `python tests/test_multi_car_sim_profile.py`
Expected: `4/4 passed`, exit 0

- [ ] **Step 10: Confirm no import breakage in precompute_scenarios**

Run: `python -c "import ast; ast.parse(open('src/simulation/multi_car_sim.py').read()); print('parse OK')"`
Then verify the backward-compat symbol still exists:
Run: `python -c "from src.simulation.multi_car_sim import COMPOUND_DEG_BASE; print(COMPOUND_DEG_BASE)"`
Expected: prints `{'SOFT': 0.09, 'MEDIUM': 0.06, 'HARD': 0.04}`

(Full `precompute_scenarios` import also pulls xgboost, which is only present in the
`f1-strategy` env — run the import check there if confirming end-to-end.)

- [ ] **Step 11: Run both Phase 0 test files together**

Run: `python tests/test_regulation_profiles.py && python tests/test_multi_car_sim_profile.py`
Expected: both report all passed, exit 0

- [ ] **Step 12: Commit**

```bash
git add src/simulation/multi_car_sim.py tests/test_multi_car_sim_profile.py
git commit -m "refactor(sim): multi_car_sim reads era constants from RegulationProfile (Phase 0)"
```

---

## Self-review

**Spec coverage (Phase 0 portion):**
- "regulation-era abstraction module + `get_profile(season)`" → Task 1. ✓
- "multi_car_sim reads constants from the profile" → Task 2. ✓
- "2022-25 profile reproduces today's exact constants" → `test_2022_25_profile_pins_legacy_constants` + `test_default_profile_matches_explicit_2022_25`. ✓
- "2026 first-class (C1-C5, override boost)" structurally present → `NEW_ERA_2026` + `test_2026_profile_drops_c6_and_changes_overtaking`. ✓ (numeric calibration is Phase 4, as scoped).
- strategy_simulator/generate_strategies refactor → explicitly deferred to Phase 4 in Scope notes (rationale: no 2022-25 behaviour change; compound-set only diverges in 2026). ✓

**Placeholder scan:** none — all code blocks complete; the only "to be calibrated later" item (2026 numeric constants) is intentional and documented, with a usable seeded default.

**Type consistency:** `RegulationProfile` field names used in `multi_car_sim` (`base_pace`, `start_fuel_kg`, `fuel_effect_per_kg`, `sc_pace_factor`, `vsc_pace_factor`, `compound_deg_base`, `compound_cliff`, `dirty_air_window`, `dirty_air_penalty`, `drs_window`, `overtake_aid_benefit`, `lap_time_noise_std`) all match the dataclass definition in Task 1. `get_profile` / `DEFAULT_PROFILE` / `GROUND_EFFECT_2022_25` names consistent across module, sim, and tests.
