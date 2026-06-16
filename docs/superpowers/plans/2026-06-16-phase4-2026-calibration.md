# Phase 4 — 2026 Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Much of this runs on the HPC **head node** (internet) — compute nodes are offline.

**Goal:** Bring 2026 into the pipeline and make current-season strategy era-correct — ingest 2026 data, append it to the committed feature parquets (without clobbering 2022–25), generate a 2026 driver config, calibrate the `new_era_2026` regulation profile from real 2026 data, and validate position + strategy on 2026.

**Architecture:** 2026 is added via `append_season.py` (merge-append, since raw 2022–25 laps aren't on disk). The `new_era_2026` profile's numeric constants (compound deg, SC priors) are calibrated by measuring 2026 data — same measure-don't-guess approach that fixed Phase 3.5. Validation reuses the Phase 2/3 harnesses (already era-aware via `get_profile(season)`).

**Tech Stack:** Python 3.11, pandas/xgboost/fastf1 (HPC). Ingestion on the **head node** (internet); everything else anywhere.

---

## Reality / constraints

- HPC head node has internet; **compute nodes do not** → ingest interactively on the head node.
- **0 raw lap files on disk** for 2022–25 → cannot re-run the full pipeline; must merge-append 2026 (that's what `append_season.py` does).
- ~7–8 races of 2026 exist so far (154 jolpica result rows) → calibration is **thin**; expect wider uncertainty. Don't overfit.

## Phase 4a — data acquisition + merge (HEAD NODE)

- [ ] **Step 1: Ingest 2026 FastF1 (laps/weather/track_status/race_control)** — the slow part (~10–30 min).
  `python -m src.ingestion.fastf1_extractor --seasons 2026`
- [ ] **Step 2: Re-fetch jolpica for all seasons** (small REST; needed because it writes single-file parquets — passing only 2026 would clobber 2022–25).
  `python -m src.ingestion.jolpica_client --seasons 2022 2023 2024 2025 2026`
- [ ] **Step 3: Re-fetch openf1 for all seasons** (small).
  `python -m src.ingestion.openf1_client --seasons 2023 2024 2025 2026`
- [ ] **Step 4: Merge 2026 into committed clean_laps + features.**
  `python -m src.preparation.append_season --season 2026`
- [ ] **Step 5: Verify 2026 landed without losing prior seasons.**
  ```bash
  python -c "import pandas as pd; s=pd.read_parquet('data/features/stint_features.parquet'); print('seasons', sorted(s.Season.unique())); print('2026 stints', int((s.Season==2026).sum()))"
  ```
  Expect seasons `[2022, 2023, 2024, 2025, 2026]` and a positive 2026 stint count.
- [ ] **Step 6: Generate + promote the 2026 driver config.**
  ```bash
  python -m src.preparation.generate_driver_configs --seasons 2026
  cp configs/generated/drivers_2026.json configs/drivers_2026.json
  ```
- [ ] **Step 7: Commit from HPC** (data produced there): the updated jolpica/openf1 parquets, 2026 weather + race_control, updated clean_laps + feature parquets, `configs/drivers_2026.json`. (Raw laps/track_status stay gitignored.) Then `git pull` on the laptop.

## Phase 4b — calibrate `new_era_2026` from data

- [ ] **Step 1: Probe 2026 vs 2022–25** (deg magnitude + compound ordering + SC rate). Reuse the DegSlope-by-compound probe restricted to 2026, and compare overall deg magnitude to 2022–25 (research says 2026 thermal deg is heavier).
- [ ] **Step 2: Set `new_era_2026` values** in `regulation_profiles.py` from the probe: `compound_deg_multiplier` from 2026 DegSlope ratios (blended with 2022–25 if 2026 is too thin), and an overall deg-magnitude note if 2026 runs hotter. Pace offset stays neutral. C1–C5 + override-boost already set.
- [ ] **Step 3: Recompute SC priors** including 2026 (`safety_car_model`), or note 2026 uses the existing priors if data is too thin.

## Phase 4c — validate on 2026

- [ ] **Step 1:** `python -m src.analysis.position_validation --seasons 2026 --n-sims 30` → 2026 finishing-order Spearman/MAE.
- [ ] **Step 2:** `python -m src.analysis.position_strategy_validation --seasons 2026 --n-sims 15` → 2026 strategy prediction (with the 2022–25 prior).
- [ ] **Step 3:** interpret with honesty about the small 2026 sample.

---

## Notes / risks

- Re-fetching jolpica/openf1 may slightly refresh historical rows — acceptable (F1 results are stable); driver-config generation + validation absorb it.
- `append_season` is idempotent (drops any prior copy of the season before concat) — safe to re-run.
- Deg model is **not** retrained for 2026 (compound-insensitive anyway); 2026 compound realism comes from the profile multiplier calibrated to 2026 DegSlope. A full blended retrain is deferred unless 2026 magnitude clearly diverges.
- Keep `n-sims` modest; 2026 has few races, so metrics are noisy — report counts alongside rates.
