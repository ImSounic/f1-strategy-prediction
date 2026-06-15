# Design: Validated, Position-Aware, Era-Correct Strategy

**Date:** 2026-06-15
**Status:** Approved (pending spec review)
**Author:** brainstorming session

---

## 1. Background & motivation

The headline strategy pipeline today is **single-car, total-time** optimization:
`strategy_simulator.run_monte_carlo` ranks candidate strategies by median race
time, and `strategy_validation_rolling` scores how often that pick matches what
the race winner actually did. It ignores **track position** entirely — undercut,
overcut, dirty air, traffic.

A full multi-car simulator already exists (`src/simulation/multi_car_sim.py`):
it models position via cumulative time, overtaking (DRS, dirty air, tyre/pace
advantage), pit position cost, blue flags, team orders, and SC field
compression. **But it is unvalidated and disconnected** — imported only by
`src/simulation/precompute_scenarios.py`, never checked against real finishing
positions, and not used by strategy selection or validation.

Two forces drive this work:
1. **Trust then use:** validate the position sim against reality, then make it
   the basis for strategy selection (choose by expected finishing position).
2. **2026 regulation reset:** the current season (mid-June 2026, ~10 races in)
   runs under physics that invalidate models trained on 2022–25 (see §8).

Prerequisite already completed this session: the validation top-k metric bug was
fixed (`src/analysis/strategy_match.py`, unit-tested) so we now have a trustworthy
stop-count vs full-strategy yardstick.

## 2. Goals

- Measure how well `multi_car_sim` predicts **real finishing positions**
  (Spearman rank correlation + position MAE), across 2022–2025.
- Make strategy selection able to choose by **expected finishing position**, and
  measure whether that beats time-optimal selection at predicting reality.
- Make the simulator **regulation-era aware** so 2026 (and future resets) are
  first-class, not retrofits.
- Calibrate a **2026 profile** so current-race strategy uses correct-era physics,
  with honest uncertainty given limited post-reset data.
- Produce **reproducible driver configs** for all seasons (replacing the
  hand-made, script-less 2024/25 files).

## 3. Non-goals

- Modeling DNFs (crashes/mechanical failures are stochastic) — validation is over
  **classified finishers only**; this caps rank-correlation below 1.0.
- Calibrating the sim's free parameters to fit data (overfitting risk). Deferred:
  only revisit if Phase 2 reveals poor baseline fidelity.
- Per-driver optimal-strategy product features beyond what validation needs.
- Wet/intermediate strategy modeling (separate future work).
- Mandatory-2-stop or other unconfirmed 2026 sporting-format rules.

## 4. Key decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Step 2 goal | Validate **and** integrate the existing multi-car sim |
| Position metric | **Spearman rank correlation + position MAE** |
| Validation scope | **All 2022–2025** (requires deriving 2022/23 configs) |
| Driver configs | **General generator for all seasons** (reproducible; replaces hand-made) |
| Validation approach | **A — full-field reconstruction** (real grid + real strategies) |
| 2026 sequencing | **Era-ready now; calibrate 2026 as Phase 4** |

## 5. Architecture — phases

### Phase 0 — Regulation-era abstraction (foundation)
New module `src/simulation/regulation_profiles.py` defining, per era:
- `compound_set` (e.g. C1–C6 for 2022–25; C1–C5 for 2026)
- `fuel_model` (`start_fuel_kg`, `burn_rate`, `fuel_effect_per_kg`)
- `overtaking_model` params (DRS-based vs override-boost; dirty-air penalty)
- `base_pace`
- `deg_model` selector (which model file / blending policy)

Plus `get_era(season) -> str` and `get_profile(season) -> RegulationProfile`.

`multi_car_sim`, `strategy_simulator`, and `generate_strategies` are refactored
to read these constants from the profile instead of module-level literals.
**The `ground_effect_2022_25` profile reproduces today's exact constants**, so
Phases 1–3 validate on unchanged physics; the abstraction only unlocks 2026.

**Interface:** `RegulationProfile` dataclass (frozen); pure, no I/O. Unit-testable.

### Phase 1 — Driver-config generator
New `src/preparation/generate_driver_configs.py`. Derives per-season ratings from
committed data, matching the documented 2024/25 methodology:
- `pace_delta` ← median per-round qualifying gap-to-pole (Q3>Q2>Q1); reference
  driver = 0.0
- `overtaking` ← mean positions gained (grid − finish) per race, normalized 0.40–0.95
- `tyre_management` ← median (driver stint length ÷ compound-median stint length),
  normalized 0.50–0.95
- `team` / `teammate` ← from results; `circuit_overtaking_difficulty` (circuit
  constant) carried through unchanged

Outputs `configs/drivers_{2022..2025}.json` in the existing schema.
**Acceptance:** regenerated 2024/25 reproduce committed values within tolerance
(document any deviation; fall back to keeping committed 2024/25 if methodology
can't be reproduced).

### Phase 2 — Position validation harness
- New `src/analysis/position_match.py` (pure, no ML deps, unit-tested):
  `score_positions(predicted_order, actual_order) -> {spearman, position_mae, n}`,
  computed over **classified finishers only**.
- New `src/analysis/position_validation.py`:
  - `reconstruct_field(season, rnd, ...)` → per classified car: grid slot
    (qualifying), real strategy (from `stint_features`, extending the existing
    winner-only reconstruction to all drivers), actual finish.
  - Reuse `load_drivers()` + `build_circuit_params()` from `precompute_scenarios`.
  - `simulate_field(...)` → run `MultiCarRaceSim` with `greedy_sc=False` (all cars
    on fixed real strategies), N seeds; mean finishing position per driver → order.
  - Aggregate per season + overall → `results/position_validation_report.json`.

### Phase 3 — Position-aware selection + integration
- `select_by_position(target, grid, field_strategies, candidates, circuit, N)`:
  for the target driver at their real grid slot with the rest of the field on
  real strategies, evaluate candidate strategies via `MultiCarRaceSim`, pick **min
  expected finishing position**.
- Integration report: for each race, compute both the **time-optimal** and
  **position-optimal** strategy for the winner and compare each to the actual
  (stop-count + full-strategy match via `strategy_match.py`). Output shows whether
  position-awareness improves strategy prediction.

### Phase 4 — 2026 calibration
- Extend ingestion config + run ingest for **2026** (FastF1/OpenF1/Jolpica support
  the live season).
- Add `new_era_2026` profile: compound set **C1–C5**; recalibrated fuel/energy
  model; **override-boost** overtaking model replacing DRS; reduced dirty-air
  penalty (−55% drag, closer following).
- **Uncertainty-aware blended degradation model**: 2022–25 model as structural
  prior, adapted with the limited 2026 data (blend / transfer), surfacing wider
  confidence bands for 2026 predictions.
- Regenerate 2026 driver config (Phase 1 generator) + recompute SC priors from
  2026 data.
- Run Phase 2 position validation on 2026.

## 6. Data flow

```
qualifying.parquet ─┐
results.parquet  ───┼─► generate_driver_configs.py ─► configs/drivers_YYYY.json
stint_features ─────┘                                        │
                                                             ▼
stint_features ──► reconstruct_field (grid + strategies) ─► position_validation.py ─► MultiCarRaceSim
pirelli csv + XGBoost deg model ──► build_circuit_params ──┘          │  (era profile drives constants)
                                                                      ▼
                                       position_match.py (Spearman + MAE, classified finishers)
                                                                      ▼
                                              results/position_validation_report.json
```

## 7. Testing

- **Pure functions, unit-tested** (pattern from `strategy_match.py`, runnable
  without the ML stack):
  - `position_match.score_positions` — synthetic orders: perfect order →
    spearman 1.0 / MAE 0; reversed → spearman −1; finisher subsetting; ties.
  - config-generator normalization helpers — known inputs → expected ranges.
  - `regulation_profiles.get_era` / profile selection — boundary seasons.
- **Smoke tests**: `MultiCarRaceSim` deterministic under a fixed seed; one-race
  reconstruction + simulation runs end-to-end and returns a full grid.
- **Reproduction check** (Phase 1): regenerated 2024/25 vs committed, within tolerance.

## 8. 2026 regulation impact (research, 2026-06-15)

| 2026 change | Affects | Action |
|---|---|---|
| Narrower tyres (F −25mm, R −30mm), new construction, **C6 dropped → C1–C5**, thermal-deg heavy → 2–3 stops | deg model, `COMPOUND_HARDNESS`, `generate_strategies` | era-aware/blended deg model; C1–C5 compound set for 2026; shift stop-count priors |
| PU ~50/50 ICE/electric, MGU-K 350kW, no MGU-H, sustainable fuel, car −30kg, **lift-and-coast** strategic | `fuel_model`, sim `base_pace`/`fuel_effect` | 2026 fuel/energy profile; recalibrate fuel effect & base pace |
| **Active aero + Override boost replaces DRS**; −55% drag, −30% downforce, closer following | `multi_car_sim` DRS/dirty-air/overtake | 2026 overtaking model (override-boost); reduce dirty-air penalty |
| Pecking-order reset (new PUs) | driver-config generator | regenerate 2026 config from 2026 data |
| SC/incident profile (new cars) | `safety_car_priors` | recompute from 2026 data |

**Sources:** [F1.com aero](https://www.formula1.com/en/latest/article/explained-2026-aerodynamic-regulations-fia-twitter-mode-z-mode-.26c1CtOzCmN3GfLMywrgb2),
[Pirelli/Motorsport C6](https://www.motorsport.com/f1/news/pirelli-sets-f1-2026-compounds-abandons-c6/10779540/),
[F1.com tyres](https://www.formula1.com/en/latest/article/pirelli-confirm-2026-tyre-compounds-as-f1-gets-set-for-a-new-era-of.6la0zKVsCYwWk9AAISz4Yw),
[The Race – 2026 terms](https://www.the-race.com/formula-1/boost-overtake-mode-active-aero-recharge-key-2026-terms-explained/).

**Constraint:** ~10 races of post-reset data across a discontinuity → 2026
predictions carry **wider uncertainty**; the value is era-correctness + honest
confidence, not 2025-level precision. Do not pool 2026 with 2022–25.

## 9. Performance

- Phase 2 ≈ 90 races × N sims × 20 cars × ~60 laps (pure-Python loops). N
  configurable; **run on HPC**. Single-car validation already took ~230s; multi-car
  is heavier per sim.
- Phase 3 is candidates × sims heavier → reduced candidate set / sims; HPC sbatch.
- Vectorization of `multi_car_sim` is a possible later optimization, not in scope.

## 10. Limitations (documented in outputs)

- No DNF modeling → finishers-only validation; rank-corr ceiling < 1.
- Driver configs are season-aggregate (no per-race form/weather).
- Grid = qualifying classification (ignores grid penalties).
- 2026 deg/overtaking calibration limited by small post-reset sample.

## 11. File layout

```
src/simulation/regulation_profiles.py        # Phase 0 (new)
src/preparation/generate_driver_configs.py   # Phase 1 (new)
configs/drivers_{2022..2025}.json            # Phase 1 (regenerated/new)
src/analysis/position_match.py               # Phase 2 (new, pure)
src/analysis/position_validation.py          # Phase 2 (new)
results/position_validation_report.json      # Phase 2 (output)
tests/test_position_match.py                 # Phase 2 (new)
tests/test_regulation_profiles.py            # Phase 0 (new)
# Phase 3: position-aware selection (extends strategy_simulator / new module)
# Phase 4: new_era_2026 profile, 2026 ingest, blended deg model, 2026 configs
Makefile: gen-configs, position-validate targets
```

## 12. Implementation note

This is an **umbrella spec across 5 phases (0–4)**. Each phase gets its own
implementation plan (via writing-plans) when started, so we never commit to one
oversized plan. Recommended order: 0 → 1 → 2 → 3 → 4.
