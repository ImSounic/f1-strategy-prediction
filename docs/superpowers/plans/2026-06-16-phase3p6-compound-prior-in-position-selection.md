# Phase 3.6 — Compound Prior in Position Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Push full-strategy (compound-sequence) prediction accuracy higher by reranking the position harness's candidates with the historical compound prior — the proven lever — while keeping it leakage-free (temporal priors).

**Architecture:** Add a third pick to `position_strategy_validation.py`: `pos+prior`, produced by reranking candidates (ordered by expected finishing position) through `CompoundPrior.rerank_strategies`. The prior is built per season from earlier seasons only (`_build_temporal_prior`, reused from `strategy_validation_rolling`). Report its stop/exact rates alongside time and position objectives.

**Tech Stack:** Python 3.11; pandas + xgboost (HPC). No new pure logic — reuses tested `rerank_strategies`.

---

## Design

- **Temporal prior (no leakage):** for season S, build the prior from seasons `< S` only. 2022 → no prior → `pos+prior` = `pos` pick.
- **Reuse `rerank_strategies`:** feed candidates sorted by `mean_pos` (best first) as `mc_results`; it blends sim rank with prior score within stop-count tiers and returns them reordered. `reranked[0]` is the prior-aware pick. Carry the original `seq` list through via a `_seq` key so it survives the rerank.
- **Reuse `_build_temporal_prior`** from `strategy_validation_rolling` (no refactor).
- The prior reranks *within* stop-count tiers, so `pos+prior` keeps the position pick's stop count and only improves the compound sequence — exactly the targeted lever.

**Success:** `pos_prior_strat_rate` > `pos_strat_rate` (0.136). Stop rate ~ unchanged.

## File structure

```
src/analysis/position_strategy_validation.py  # MODIFY — add temporal prior + pos+prior pick
```

---

## Task 1: Add the prior-aware pick to the harness

**Files:**
- Modify: `src/analysis/position_strategy_validation.py`

- [ ] **Step 1: Add imports**

After the existing imports, add:

```python
from src.simulation.compound_prior import CompoundPrior
from src.analysis.strategy_validation_rolling import _build_temporal_prior
```

- [ ] **Step 2: Build a temporal prior per season and pass it into `evaluate_race`**

In `run(...)`, inside the `for season in seasons:` loop, BEFORE the `for rnd in rounds:` loop, add:

```python
        prior_seasons = [s for s in (2022, 2023, 2024, 2025) if s < season]
        prior = None
        if prior_seasons:
            prior = _build_temporal_prior(
                Path("data/features"),
                Path(raw["supplementary"]) / "pirelli_circuit_characteristics.csv",
                Path(raw["jolpica"]) / "results.parquet",
                prior_seasons,
            )
            logger.info(f"  [{season}] compound prior from seasons {prior_seasons}")
```

Then change the `evaluate_race(...)` call to pass `prior`:

```python
            row = evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile, prior)
```

- [ ] **Step 3: Extend `evaluate_race` to compute the prior pick**

Change the signature:
```python
def evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile):
```
to:
```python
def evaluate_race(season, rnd, field, drivers, circuit, n_sims, profile, prior=None):
```

After `pos_pick = argmin_by(stats, "mean_pos")`, add:

```python
    # Position + compound prior: rerank candidates (best position first) through
    # the historical prior, which fixes compound choice within the stop tier.
    pos_prior_pick = pos_pick
    if prior is not None and stats:
        ranked = sorted(stats, key=lambda d: d["mean_pos"])
        mc_results = [{
            "compound_sequence": " → ".join(d["seq"]),
            "num_stops": d["num_stops"],
            "median_time": d["mean_pos"],
            "_seq": d["seq"],
        } for d in ranked]
        reranked = prior.rerank_strategies(mc_results, circuit.circuit_key, blend_weight=0.3)
        if reranked:
            top = reranked[0]
            pos_prior_pick = {"seq": top["_seq"], "num_stops": top["num_stops"]}
```

After the existing `p_stop, p_exact = _score_pick(pos_pick, real)`, add:

```python
    pp_stop, pp_exact = _score_pick(pos_prior_pick, real)
```

In the returned dict, add these keys (after the `pos_*` keys):

```python
        "pos_prior_pick_seq": pos_prior_pick["seq"],
        "pos_prior_stop_match": pp_stop, "pos_prior_strat_exact": pp_exact,
```

- [ ] **Step 4: Add prior rates to `_rates`**

In `_rates`, add two lines to the returned dict:

```python
        "pos_stop_rate": rate("pos_stop_match"),
        "pos_strat_rate": rate("pos_strat_exact"),
        "pos_prior_stop_rate": rate("pos_prior_stop_match"),
        "pos_prior_strat_rate": rate("pos_prior_strat_exact"),
    }
```

- [ ] **Step 5: Update the per-race log line (optional clarity)**

Change the `logger.info` in `run` that prints the per-race result to also show the prior pick:

```python
            logger.info(f"  {season} r{rnd:>2} {ckey:<14} "
                        f"time[s={row['time_stop_match']:d} e={row['time_strat_exact']:d}] "
                        f"pos[s={row['pos_stop_match']:d} e={row['pos_strat_exact']:d}] "
                        f"pos+prior[s={row['pos_prior_stop_match']:d} e={row['pos_prior_strat_exact']:d}]")
```

- [ ] **Step 6: Syntax-check**

Run: `python -m py_compile src/analysis/position_strategy_validation.py`
Expected: no output, exit 0

- [ ] **Step 7 (HPC): Smoke run**

Run: `python -m src.analysis.position_strategy_validation --seasons 2024 --n-sims 5`
Expected: per-race lines now show `pos+prior[...]`; season `_rates` includes `pos_prior_*`. (2024's prior comes from 2022-23.)

- [ ] **Step 8 (HPC): Full run + read the gain**

Run: `make position-strategy`
Then:
```bash
python -c "import json; r=json.load(open('results/position_strategy_report.json'))['overall']; print(r)"
```
Compare `pos_prior_strat_rate` vs `pos_strat_rate` (0.136). Success = clear lift.

- [ ] **Step 9: Commit** (skipped — user commits at end)

---

## Self-review

**Spec coverage:** adds the compound prior (proven lever) to position selection, leakage-free (temporal), reported as a third objective for clean measurement. ✓
**Placeholder scan:** none — full code/edits given.
**Type consistency:** `_build_temporal_prior(features_dir, circuit_csv, results_path, train_seasons)` and `CompoundPrior.rerank_strategies(mc_results, circuit_key, blend_weight)` match their definitions; `_seq` carried through rerank; `circuit.circuit_key` is a `CircuitParams` field; `_score_pick(pick, real)` expects `{seq, num_stops}` which `pos_prior_pick` provides.
