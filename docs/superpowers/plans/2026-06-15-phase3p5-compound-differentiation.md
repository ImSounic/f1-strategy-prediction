# Phase 3.5 — Compound Pace Differentiation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the multi-car sim a realistic compound tradeoff (SOFT faster fresh but degrades faster; HARD slower but durable) so strategy selection stops defaulting to SOFT and can match real compound choices.

**Architecture:** Add two era-specific fields to `RegulationProfile` — `compound_pace_offset` (s/lap, by compound) and `compound_deg_multiplier` (×, by compound) — and apply them in `multi_car_sim` (`_compute_lap_time` adds the offset; `_get_deg_rate` applies the multiplier). Re-validate Phases 2 & 3.

**Tech Stack:** Python 3.11; numpy (sim). Tests are plain-Python / numpy, runnable on the laptop.

---

## Rationale

The Phase 3 dump showed every pick starting on SOFT because the sim can't distinguish compounds: no base-pace-per-compound offset, and the XGBoost deg model is compound-insensitive (`CompoundHardness` SHAP ≈ 0.0001). Adding a pace offset *and* a degradation multiplier restores the pace-vs-durability tradeoff.

**This intentionally changes 2022–25 sim behaviour** (the Phase 0 "no behaviour change" guarantee is deliberately superseded here). The gate is re-validation: Phase 2 Spearman must hold (~0.69), Phase 3 compound-exact should improve off ~0.

Default values (literature-typical; tune if re-validation says so):
- `compound_pace_offset = {SOFT: -0.7, MEDIUM: 0.0, HARD: 0.5}`
- `compound_deg_multiplier = {SOFT: 1.6, MEDIUM: 1.0, HARD: 0.65}`

Fallback caveat: when `circuit.deg_rates` is absent the sim falls back to the (already differentiated) `compound_deg_base`, and the multiplier double-counts slightly — harmless and rarely hit (production always supplies XGBoost deg rates).

## File structure

```
src/simulation/regulation_profiles.py    # MODIFY — two new profile fields (both eras)
src/simulation/multi_car_sim.py          # MODIFY — apply offset + multiplier
tests/test_regulation_profiles.py        # MODIFY — assert new fields
tests/test_multi_car_sim_profile.py      # MODIFY — assert compound tradeoff
```

---

## Task 1: Add compound fields to the profile

**Files:**
- Modify: `src/simulation/regulation_profiles.py`
- Test: `tests/test_regulation_profiles.py`

- [ ] **Step 1: Extend the failing test**

In `tests/test_regulation_profiles.py`, add these two tests (after the existing ones, before `_run_all`):

```python
def test_profiles_have_compound_pace_and_deg_fields():
    for p in (GROUND_EFFECT_2022_25, NEW_ERA_2026):
        assert set(p.compound_pace_offset) == {"SOFT", "MEDIUM", "HARD"}
        assert set(p.compound_deg_multiplier) == {"SOFT", "MEDIUM", "HARD"}
        # SOFT faster fresh than HARD; SOFT degrades more than HARD
        assert p.compound_pace_offset["SOFT"] < p.compound_pace_offset["HARD"]
        assert p.compound_deg_multiplier["SOFT"] > p.compound_deg_multiplier["HARD"]
        assert p.compound_pace_offset["MEDIUM"] == 0.0
        assert p.compound_deg_multiplier["MEDIUM"] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_regulation_profiles.py`
Expected: FAIL — `AttributeError: 'RegulationProfile' object has no attribute 'compound_pace_offset'`

- [ ] **Step 3: Add the fields to the dataclass + both eras**

In `src/simulation/regulation_profiles.py`, add two fields to the dataclass (after `compound_cliff`):

```python
    compound_cliff: dict        # {compound_name: cliff lap}
    compound_pace_offset: dict  # {compound_name: s/lap vs MEDIUM (SOFT faster)}
    compound_deg_multiplier: dict  # {compound_name: x base deg (SOFT degrades more)}
```

In `GROUND_EFFECT_2022_25`, add (after `compound_cliff=...`):

```python
    compound_cliff={"SOFT": 20, "MEDIUM": 30, "HARD": 40},
    compound_pace_offset={"SOFT": -0.7, "MEDIUM": 0.0, "HARD": 0.5},
    compound_deg_multiplier={"SOFT": 1.6, "MEDIUM": 1.0, "HARD": 0.65},
```

In `NEW_ERA_2026`, add the same two lines after its `compound_cliff=...` (calibrated in Phase 4):

```python
    compound_cliff={"SOFT": 20, "MEDIUM": 30, "HARD": 40},
    compound_pace_offset={"SOFT": -0.7, "MEDIUM": 0.0, "HARD": 0.5},
    compound_deg_multiplier={"SOFT": 1.6, "MEDIUM": 1.0, "HARD": 0.65},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python tests/test_regulation_profiles.py`
Expected: all pass (now 7 tests)

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: Apply compound effects in the simulator

**Files:**
- Modify: `src/simulation/multi_car_sim.py`
- Test: `tests/test_multi_car_sim_profile.py`

- [ ] **Step 1: Extend the behaviour test**

In `tests/test_multi_car_sim_profile.py`, add a single-car helper + two tests (after the existing tests, before `_run_all`):

```python
def _solo_time(compound, laps, seed=42):
    drivers = [DriverConfig("AAA", "A", "t1", 0.0, 0.5, 0.7, "")]
    circuit = CircuitParams(
        circuit_key="t", circuit_name="T", total_laps=laps,
        pit_loss_seconds=20.0, sc_prob_per_race=0.0, vsc_prob_per_race=0.0,
        overtaking_difficulty=0.5, deg_rates={},
    )
    strat = Strategy(stints=[(compound, laps)], name=compound)
    return MultiCarRaceSim(circuit, drivers, [strat], 0, strat,
                           greedy_sc=False).run(seed=seed)["target_time"]


def test_soft_faster_when_fresh_short_race():
    # Over a short stint the pace offset dominates -> SOFT quicker than HARD.
    assert _solo_time("SOFT", 8) < _solo_time("HARD", 8)


def test_hard_faster_over_long_stint():
    # Over a long stint degradation dominates -> HARD quicker than SOFT.
    assert _solo_time("HARD", 45) < _solo_time("SOFT", 45)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_multi_car_sim_profile.py`
Expected: `test_soft_faster_when_fresh_short_race` and/or `test_hard_faster_over_long_stint` FAIL (no compound differentiation yet — times nearly equal).

- [ ] **Step 3: Apply the pace offset in `_compute_lap_time`**

In `src/simulation/multi_car_sim.py`, in `_compute_lap_time`, change the base+driver line:

```python
        # Base + driver delta
        lap_time = self.profile.base_pace + driver.pace_delta
```
to:
```python
        # Base + driver delta + compound pace offset (SOFT faster fresh)
        lap_time = self.profile.base_pace + driver.pace_delta
        lap_time += self.profile.compound_pace_offset.get(car.tyre_compound, 0.0)
```

- [ ] **Step 4: Apply the deg multiplier in `_get_deg_rate`**

Change:
```python
        base = circuit.deg_rates.get(compound, self.profile.compound_deg_base[compound])
        # Better tyre management (higher rating) = lower deg
        driver_factor = 1.0 + 0.3 * (1.0 - driver.tyre_management)
        return base * driver_factor
```
to:
```python
        base = circuit.deg_rates.get(compound, self.profile.compound_deg_base[compound])
        # Compound relativity: SOFT degrades faster, HARD slower (the XGBoost deg
        # model is compound-insensitive, so we inject the relative ordering here).
        base *= self.profile.compound_deg_multiplier.get(compound, 1.0)
        # Better tyre management (higher rating) = lower deg
        driver_factor = 1.0 + 0.3 * (1.0 - driver.tyre_management)
        return base * driver_factor
```

- [ ] **Step 5: Run to verify it passes**

Run: `python tests/test_multi_car_sim_profile.py`
Expected: all pass (now 6 tests), including the two compound-tradeoff tests.

- [ ] **Step 6: Run all sim/profile tests together**

Run: `python tests/test_regulation_profiles.py && python tests/test_multi_car_sim_profile.py`
Expected: both fully pass.

- [ ] **Step 7: Commit** (skipped — user commits at end)

---

## Task 3: Re-validate (the gate)

**Files:** none (validation run).

- [ ] **Step 1 (HPC): Re-run Phase 2 + Phase 3**

```bash
make position-validate
make position-strategy
python -c "import json; print('POS-VALID', json.load(open('results/position_validation_report.json'))['overall']); print('POS-STRAT', json.load(open('results/position_strategy_report.json'))['overall'])"
```

- [ ] **Step 2: Check the gate**

- Phase 2 `mean_spearman` should **hold ~0.69** (compound realism shouldn't hurt finishing-order accuracy; ideally helps slightly).
- Phase 3 `pos_strat_rate` / `time_strat_rate` should **rise off ~0** (the picks now use real compounds). `pos_stop_rate` should stay ≥ `time_stop_rate`.
- If Spearman regresses materially (> ~0.05 drop) or compound-exact stays ~0, the default offset/multiplier need tuning — adjust the two dicts in `regulation_profiles.py` and re-run.

- [ ] **Step 3: Commit** (skipped — user commits regenerated reports)

---

## Self-review

**Spec coverage:** addresses the Phase 3 finding (always-SOFT) by adding both missing dimensions (pace + deg) era-specifically; re-validation gate included. ✓
**Placeholder scan:** none — full code + exact values.
**Type consistency:** new fields `compound_pace_offset` / `compound_deg_multiplier` added to the dataclass and both era instances; referenced in `multi_car_sim` via `self.profile.compound_pace_offset` / `self.profile.compound_deg_multiplier`, matching names exactly.
