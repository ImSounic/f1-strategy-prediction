# Design: RL-3 — Evaluation & Showcase

**Date:** 2026-06-19
**Status:** Approved (design); spec for review
**Parent:** RL-2 league (`models/rl_league/league/`, trained main agents beat anchors ~85%)

---

## 1. Background & goal

RL-2 produced a trained AlphaStar-style league. RL-3 turns the trained `main` agent into
evidence and showcase artifacts for the personal website (existing Next.js app in
`frontend/`, deployed on Vercel). Three pillars (all in scope, maximal):

1. **Beat baselines in-sim** — rigorously, vs the scripted anchors AND the project's Phase-3
   position+prior strategy picker, with confidence intervals.
2. **Real-scenario adaptation** — a broad counterfactual sensitivity battery showing the
   agent responds to conditions (safety car, start compound, grid slot, SC probability).
3. **Emergent behaviour** — strategy fingerprint, tactic detection (undercut/overcut, SC
   opportunism), and policy introspection (P(pit) over race state).

Plus a **dashboard** integrated into the existing `frontend/` site.

## 2. Approach: standalone policy + validated sim + pluggable controllers

Evaluation does NOT use RLlib's training machinery. We load the trained `main` module from
the checkpoint (validated `from_checkpoint`), wrap its forward pass as an obs→action
function, and run races on the **validated `multi_car_sim`** where every car is driven by a
pluggable **controller**:

- `RLController` — `ma_obs.build_obs(state)` → module forward → legal greedy action.
- `ScriptedController` — a fixed pit plan (the anchors).
- `Phase3Controller` — the Phase-3 position+prior chosen strategy expressed as a plan.

This reuses all validated pieces (`multi_car_sim`, `ma_obs`, trained weights), decouples eval
from training (fast, deterministic, full control of the mixed field), and makes baselines
first-class opponents. Stats + scripted-controller logic are pure → laptop-testable; only the
model forward + sim runs on HPC. (Rejected: `Algorithm.evaluate()` — heavier, less control
over a custom mixed field.)

## 3. Decomposition (each its own spec→plan→build; RL-3a first)

- **RL-3a — Eval harness + beat-baselines** *(detailed below; build first)*
- **RL-3b — Counterfactual sensitivity battery** (force-SC-at-lap sim hook + start compound /
  grid slot / SC-probability sweeps; adaptation figures + JSON)
- **RL-3c — Emergent behaviour + introspection** (strategy fingerprint vs baselines + real
  2025; undercut/overcut & SC-opportunism detection; P(pit) heatmaps over lap×tyre-age)
- **RL-3d — Website integration** (render the RL-3a/b/c JSON in the existing `frontend/`
  Next.js app — charts, counterfactual explorer, fingerprints — and redeploy on Vercel)

## 4. Data contract (cross-cutting)

RL-3a/b/c each emit a **stable, versioned JSON** (metrics + CIs + per-condition results) plus
static figures. JSON is the single source of truth the website consumes. Target location:
`results/rl_eval/<section>.json` (+ figures under `results/rl_eval/figures/`); RL-3d wires
these into `frontend/` (copy into `frontend/public/rl/` at build, or import directly). Schema
is defined per section in each sub-spec; every metric ships with its sample size and 95% CI.

---

## 5. RL-3a — Eval harness + beat-baselines (detailed)

### 5.1 Components / files
- `src/rl/eval/__init__.py`
- `src/rl/eval/controllers.py` — `Controller` protocol with `decide(state, car_idx) -> action_int`;
  `ScriptedController(plan)` (pure); `RLController(module)` (forward + `legal_action_mask`).
- `src/rl/eval/race_runner.py` — `run_race(circuit, drivers, controllers, seed) -> RaceResult`
  (finishing positions per car). Drives `multi_car_sim` reset/step/results with a per-lap
  `pit_override` built from each controller's `decide(...)` (mirrors the RL env loop, no Ray).
- `src/rl/eval/baselines.py` — build anchor controllers (1-stop, 2-stop) and the
  `Phase3Controller` (load the Phase-3 chosen plan from the position+prior selection /
  precomputed `results/`), as plans.
- `src/rl/eval/model_loader.py` — `load_main_module(checkpoint, module_id="main_1")` →
  inference module (via validated `Algorithm.from_checkpoint` then `get_module`).
- `src/rl/eval/metrics.py` — **pure**: `win_rate(a_finishes, b_finishes)`, `mean_finish`,
  `bootstrap_ci(values, n=10000)`, head-to-head aggregation.
- `src/rl/eval/run_beat_baselines.py` — orchestrate: load module → assemble a field
  (RL protagonist on a subset of grid slots + baseline opponents) → run N seeded races →
  compute per-matchup win-rate & mean-finish with 95% CIs → write
  `results/rl_eval/beat_baselines.json` + figures.
- `scripts/hpc/eval_rl.sbatch`.
- Tests (pure, laptop): `tests/test_eval_metrics.py` (win-rate, bootstrap CI determinism),
  `tests/test_eval_controllers.py` (`ScriptedController` fires the plan; `pit_override` shape).

### 5.2 Protocol detail
`run_race` each lap: for every car, `controllers[i].decide(state_i, i)` → action int →
`action_to_compound` under `legal_action_mask` → `pit_override[i]`. Then `sim.step(pit_override)`.
At done, return `[(controller_label, finish_pos)]`. The RL protagonist uses greedy actions
(argmax), opponents follow their plans. Same `build_obs`/mask path as the validated env.

### 5.3 Field composition for the headline result
Each race: 1 RL `main` car + a field mixing the baselines (anchors + Phase-3 picker) across
the other grid slots, with driver attributes held fixed per slot so position reflects
strategy, not car pace. Rotate the RL car's grid slot across seeds to remove slot bias.
Pairwise win-rate = fraction of races the RL car finishes ahead of each baseline type.

### 5.4 Gate (RL-3a success)
`results/rl_eval/beat_baselines.json` + a figure showing, over ≥200 seeded races: RL `main`
vs `anchor_onestop`, `anchor_twostop`, `phase3_picker` — mean finishing position and
head-to-head win-rate each with 95% CIs, with the RL agent significantly ahead (CI separation)
of at least the anchors, and reported honestly vs Phase-3 (whatever the result).

### 5.5 Testing
- Pure (laptop): metrics (win-rate values, bootstrap CI reproducible with seed), controller
  plan firing, `pit_override` construction.
- HPC: a smoke (`--races 4`) runs the full field and writes a well-formed JSON; then the real
  `--races 200+`.

---

## 6. RL-3b — Counterfactual battery (outline)
Add a deterministic **force-SC-at-lap** hook to `multi_car_sim` (and expose start-compound +
grid via controllers/config). Sweeps: SC at laps {early, mid, late} × {SC, no-SC}; start
compound {SOFT, MED, HARD}; grid slot {P1, P10, P20}; SC-probability levels. For each, record
the RL agent's pit-lap / compound response and finishing delta → `results/rl_eval/
counterfactuals.json` + figures (e.g. pit-lap vs SC-lap curve). Gate: the agent's response
shifts sensibly (e.g. pits within a few laps of an injected SC).

## 7. RL-3c — Emergent behaviour + introspection (outline)
From many RL races: stop-count & compound-sequence distributions vs anchors/Phase-3/real 2025
(strategy fingerprint); undercut/overcut detection (pit-before-rival then finish-ahead) and
SC-opportunism rate; P(pit) heatmap by sampling the policy over a lap×tyre-age grid (greedy +
prob). → `results/rl_eval/emergent.json` + figures. Gate: fingerprint + at least one clearly
identified tactic + a readable decision-boundary heatmap.

## 8. RL-3d — Website integration (outline)
Consume the three JSON files in the existing `frontend/` Next.js app: a results section with
the beat-baselines chart, an interactive counterfactual explorer, and the strategy
fingerprint / heatmap. Copy JSON+figures into `frontend/public/rl/` (or import). Redeploy on
Vercel. Own spec; uses the Vercel/Next.js skills. Depends on a/b/c JSON existing.

## 9. Risks
- **Phase3Controller fidelity** — must faithfully express the Phase-3 selection as an in-sim
  plan; if the Phase-3 pipeline output isn't directly a stint plan, add a thin adapter (and
  state the assumption). The honest comparison depends on this being right.
- **Model loading** — `from_checkpoint` is validated; extracting one module for inference is
  the new bit (guard + smoke).
- **Compute** — eval is cheap vs training (no learner); 200+ races is minutes on CPU workers.

## 10. File layout (RL-3a)
```
src/rl/eval/{__init__,controllers,race_runner,baselines,model_loader,metrics,run_beat_baselines}.py
scripts/hpc/eval_rl.sbatch
tests/test_eval_metrics.py
tests/test_eval_controllers.py
results/rl_eval/beat_baselines.json   # OUTPUT (+ figures/)
```

## 11. Decomposition note
RL-3a is built in two layers: the **pure core** (`metrics.py`, `ScriptedController`,
`pit_override` logic) — laptop-tested — and the **HPC layer** (`model_loader`, `RLController`
forward, `run_beat_baselines`). Plan sequences pure before HPC, as in RL-2.
