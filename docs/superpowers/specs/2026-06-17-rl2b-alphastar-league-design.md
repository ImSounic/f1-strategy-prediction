# Design: RL-2b — AlphaStar League Layer

**Date:** 2026-06-17
**Status:** Approved (design); spec for review
**Parent:** `docs/superpowers/specs/2026-06-16-rl2-league-selfplay-training-design.md` (RL-2 umbrella)

---

## 1. Background

RL-2a proved the PPO pipeline on the validated env: a single shared `main` policy
self-played on Bahrain and converged (`episode_return_mean` → 0.0 by ~iter 11). That flat
zero is the expected dead-end of pure self-play — with every car running the same policy,
positions-gained is zero-sum and carries no "beat the field" signal. RL-2b adds a **full,
scaled AlphaStar-style league** so the learner faces *non-identical* opponents (frozen
snapshots, scripted anchors, exploiters) and the reward becomes informative again. This is
where emergent strategy (undercut/overcut, SC reactivity, defence) is expected to appear.

## 2. Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Field composition | **Mixed field, train on all learner cars** (max sample efficiency; natural multiplayer mapping) |
| Population | **Scaled: 2 main + 2 main-exploiter + 2 league-exploiter** (6 learning policies; counts configurable) + growing frozen-snapshot pool + fixed scripted anchors |
| Build approach | **One RLlib `Algorithm`, many policies**; `policies_to_train` = learners only; league controlled via callbacks + `algo.add_policy` |
| "Match" / win | one 21-car race; pairwise win = car-A finishes ahead of car-B |
| Reward | unchanged — terminal positions-gained + DSQ −50 (RL-1) |
| Scope | single circuit (Bahrain) for RL-2b; multi-circuit deferred to RL-2c |
| Compute | 1 training sbatch (~6× RL-2a learner cost); 2nd sbatch = concurrent eval vs anchors |

## 3. Components

### 3.1 Populations & policy IDs
- Learning (in `policies_to_train`): `main_0..main_{Nm-1}`, `mexp_0..`, `lexp_0..`
  (defaults Nm=Nme=Nle=2 → 6 learners).
- Frozen (not trained): `snap_{k}` (added at runtime), `anchor_{name}` (fixed set).
- Scripted anchors: validated/standard strategies — the Phase-3 position+prior pick, a
  canonical 1-stop, and a canonical 2-stop — as the absolute yardstick.

### 3.2 Per-race field (mixed, 21 cars)
- Each race picks a rotated **training-focus** policy `F` (cycles over the 6 learners so each
  gets ample data across races).
- ~half the grid (configurable `focus_share`, default ≈0.5) is assigned `F`; the remaining
  cars are **opponents** drawn by `F`'s role rule:
  - `main` → PFSP over {snapshots ∪ anchors ∪ other mains}, hard-weighted, with a self-play fraction.
  - `mexp` (main-exploiter) → opponents = current `main` agents.
  - `lexp` (league-exploiter) → PFSP over the whole league.
- **Train on every learner-controlled car** present (F's cars + any learner opponents).

### 3.3 Win-rate matrix + PFSP (pure, unit-testable)
- `WinRateMatrix`: running counts `(wins, games)` per ordered policy pair; `rate(A,B)` =
  fraction of A-car-vs-B-car pairings where A finished ahead. Updated from the full finishing
  order each race (all co-present pairs).
- PFSP sampler: opponent B drawn for learner A with prob ∝ `f(rate(A,B))`.
  - `f_hard(x) = (1−x)**p` (default p=2) — focus on opponents A loses to / is even with.
  - mixed with weight `eps` uniform over the candidate pool to guarantee coverage.
- These are pure functions over plain dicts/lists → laptop tests (no Ray).

### 3.4 League controller (RLlib callbacks)
- `on_episode_end`: read finishing order + each car's policy id → update `WinRateMatrix`.
- `on_train_result`:
  - **Snapshot add**: when a `main` agent's win-rate vs the league ≥ `snapshot_threshold`
    (default 0.70) OR every `snapshot_every` env-steps → freeze a copy as a new `snap_{k}`
    via `algo.add_policy` (frozen: not in `policies_to_train`).
  - **Exploiter reset**: when an `mexp`/`lexp` beats its target ≥ `exploiter_threshold`
    (0.70) OR exceeds `exploiter_max_steps` → snapshot it into the pool, then reinit its
    weights (fresh hunt). Cadences/thresholds configurable.
- The controller's *decisions* (what to snapshot/reset, opponent sampling) are pure functions;
  the callback is a thin adapter that calls them and performs the RLlib side-effects.

### 3.5 Scripted anchors
- Tiny custom RLModule per anchor: deterministic action from a fixed strategy plan
  (pit laps + compounds) given the current lap/tyre-age in the obs. Frozen; race in-env as
  opponents; never trained.

## 4. Data flow
race rollout → `on_episode_end` records (policy_id, finish) per car → `WinRateMatrix` →
PFSP sampler informs next races' opponent draw (via `policy_mapping_fn` reading the current
assignment) → `on_train_result` manages snapshots/resets → checkpoints.

## 5. Reward & legality
Unchanged from RL-1: `terminal_reward(grid, finish, used_two_compounds)` → positions-gained,
or `DSQ_PENALTY=−50` for a single-compound dry race. Action legality masking unchanged.

## 6. Compute & ops
- One training Algorithm with ~6 learners updated per iteration → ≈6× RL-2a's learner time;
  mitigate with `num_epochs` (default 10), worker count, long `--time`, checkpoint-resume.
- 2nd sbatch runs periodic **evaluation** of `main` agents vs the scripted anchors from
  checkpoints (feeds RL-3), in parallel with training.

## 7. Testing
- **Pure (laptop):** `WinRateMatrix` update/rate; PFSP weighting (hard-weight monotonicity,
  eps coverage, degenerate cases); role-based opponent selection (mexp targets mains, etc.);
  field assembly (focus_share, anchors always eligible); snapshot/reset trigger predicates.
- **HPC smoke:** `train_league.py --iters 2` builds the multi-policy Algorithm, runs, records
  win-rates, performs one `add_policy`, checkpoints — no errors.
- **HPC gate:** see §8.

## 8. RL-2b gate
- `main` win-rate vs scripted anchors **climbs and holds > 50%** (target 60–70%).
- Snapshot pool grows over training; exploiters reset at least once.
- Training stable (no policy collapse / NaNs).
- Sanity rollout of a `main` agent shows strategic variety (not a single fixed plan):
  responds to SC, varies pit timing/compounds vs different opponents.

## 9. Risks (honest)
- **Runtime `algo.add_policy` / weight reinit on the 2.55 new-API stack** is the top risk —
  API may differ; isolate behind pure decision functions so only the thin callback adapter
  needs HPC iteration.
- Scaled populations are compute-heavy; convergence may take many hours — checkpoint-resume
  is mandatory.
- League instability (cycling, exploiter dominance) — PFSP weighting, thresholds, and reset
  cadence will need tuning; the win-rate matrix + eval-vs-anchors make this observable.

## 10. File layout
```
src/rl/league.py             # NEW — roles/config, WinRateMatrix, PFSP sampler, field assembly,
                             #       snapshot/reset predicates (PURE; laptop-tested)
src/rl/scripted_anchors.py   # NEW — deterministic anchor RLModule(s) from fixed strategy plans
src/rl/league_callbacks.py   # NEW — RLlib callbacks: record results, manage league (thin adapter)
src/rl/train_league.py       # NEW — assemble multi-policy PPOConfig + train loop + checkpoint
scripts/hpc/train_league.sbatch          # NEW — training job (resumable)
scripts/hpc/eval_league.sbatch           # NEW — concurrent eval vs anchors (2nd sbatch)
tests/test_league.py         # NEW — pure tests (matrix, PFSP, role/field, predicates)
models/rl_league/league/     # OUTPUT — checkpoints (gitignored)
```

## 11. Decomposition note
RL-2b is built in two layers: **(a)** the pure league core (`league.py` + tests) — fully
laptop-verifiable; **(b)** the RLlib integration (`scripted_anchors.py`,
`league_callbacks.py`, `train_league.py`, sbatch) — HPC-verified. The plan sequences (a)
before (b) so the league logic is proven before wiring it into Ray.
