# Design: SOTA Multi-Agent Self-Play RL for F1 Strategy

**Date:** 2026-06-16
**Status:** Approved (pending spec review)

---

## 1. Background & goal

The existing RL (`src/rl/`) is **single-car**: `F1StrategyEnv` races one car against the
clock — no competitors, no track position, binary stay/pit action, auto-compound,
reward = per-lap shaping + terminal *total time*. It's disconnected from the
multi-car position work we validated this session.

**Goal:** a state-of-the-art **multi-agent, league self-play** RL agent that learns
pit strategy (timing **and** compound) by racing against other learning agents in
the **validated** multi-car simulator, optimizing **finishing position**. Success is
three-fold:
1. **Beat our baselines in-sim** — higher expected finishing position than the
   Phase-3 picks (time-optimal, position-optimal, position+prior) in `multi_car_sim`.
2. **Real-world usable** — strategies that would help a real team (credible because
   the training physics are the validated sim, Spearman 0.71 vs real results).
3. **Emergent strategy** — undercut/overcut, reactive SC pitting, defensive
   track-position play arising from self-play, not hand-coded.

## 2. Key decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Framework | **Ray RLlib** (full MARL: league play, shared/independent policies) |
| Policy | **One shared, driver-conditioned policy** (driver attrs in obs → drives any car) |
| Self-play | **League / opponent-pool** (AlphaStar-style snapshot pool + scripted strategies) |
| Action | **Discrete(4)**: stay / pit-Soft / pit-Medium / pit-Hard (+ legality masking) |
| Reward | mostly **terminal = positions gained (grid − finish)**, minimal shaping |
| Env physics | **Approach A** — refactor `multi_car_sim` into a step-able core shared by the
existing fixed-strategy `run()` AND the RL env (one validated source of truth) |
| Eval | triple: in-sim vs baselines · real-scenario counterfactuals · emergent-behaviour analysis |

**Why Approach A (not a separate env):** RL optimizes for whatever physics it trains
in, and exploits sim bugs (reward hacking). Training in a *separate* env then scoring
in `multi_car_sim` is a train/test mismatch that invalidates criterion 1, and an
unvalidated env invalidates criterion 2. A shares the validated physics; a regression
test (fixed-strategy stepping must reproduce `run()`) gives the bug-catch a second
implementation would, without divergence or double maintenance.

## 3. Decomposition (4 sub-projects, each its own spec→plan→build)

- **RL-1 — Multi-agent environment** *(this spec details it; build first)*
- **RL-2 — League self-play training** (RLlib PPO, opponent pool, HPC)
- **RL-3 — Evaluation triple** (baselines / real counterfactual / emergence)
- **RL-4 — (optional) serving/recommender** integration

Each lands and is validated before the next. RL-1 is the foundation everything rests on.

---

## 4. RL-1 — Multi-agent environment (detailed)

### 4.1 Step-able physics core (refactor)
Refactor `MultiCarRaceSim` so the per-lap logic lives in `reset(seed)` + `step(pit_actions)`:
- `step(pit_actions: dict[int, int])` advances **one lap** for the whole field
  (SC/VSC update, field compression, gaps, per-car pit + lap-time + cumulative,
  position update, lapped check, overtaking) using the existing methods.
- `run()` becomes: `reset()` then loop `{build fixed-strategy pit_actions; step(...)}`
  — so existing behaviour (and Phase 2/3 validation) is preserved **by construction**
  (same code path), not by a parallel reimplementation.
- **Regression guard:** a golden test (fixed scenario + seed → stable
  `finishing_positions`/`target_time`) locks the refactor; and we **re-run Phase 2/3
  validation** afterwards — Spearman must stay ≈0.70 (the real proof physics didn't drift).

### 4.2 RLlib `MultiAgentEnv`
`src/rl/multiagent_env.py` — wraps the step-able sim:
- **Agents:** the N classified cars (one shared policy; `policy_mapping_fn` → "shared").
- **Observation (per car), driver-conditioned:** race progress, own tyre compound/age,
  cumulative deg, position, gap-ahead, gap-behind, fuel, SC/VSC state, stops done,
  compounds used, laps-since-SC, **driver attributes (pace_delta, overtaking,
  tyre_management)**, circuit constants (pit loss, SC prob, overtaking difficulty).
- **Action:** `Discrete(4)` → {stay, pit-Soft, pit-Medium, pit-Hard}, translated to the
  sim's per-car pit + compound. **Legality masking** (no lap-1/late pit, min stint ≥ a
  few laps, ≤ max_stops); illegal → treated as stay. 2-compound FIA rule enforced via a
  terminal penalty if violated.
- **Reward:** terminal per car = **(grid_position − finish_position)** (positions gained),
  optional tiny per-lap shaping (kept minimal to preserve emergence); illegal-strategy
  penalty. SC/VSC laps contribute no shaping (out of driver control).
- **Era-aware:** uses `get_profile(season)` like the rest of the pipeline.

### 4.3 Pure, unit-testable helpers (laptop-runnable, no Ray/torch)
`src/rl/ma_obs.py` (or similar): `build_obs(car_state, field, driver, circuit)`,
`reward_from_positions(grid, finish)`, `legal_actions(car_state, rules)`,
`action_to_pit(action)` — pure functions, unit-tested without the heavy stack.

### 4.4 Tests
- Pure helpers: obs bounds/shape, reward sign (gain→+), masking correctness,
  action→compound mapping. (laptop)
- Env: RLlib `check_env`/API conformance; one fixed-policy episode runs; reward only
  at terminal; masked actions never pit illegally. (HPC, needs ray)
- Physics regression: golden `run()` output stable; **Phase 2/3 Spearman unchanged**. (HPC)

---

## 5. RL-2 — League self-play (outline)
RLlib PPO, shared policy. **League/opponent pool:** maintain a growing set of policy
snapshots (+ the scripted/real strategies as fixed opponents); `policy_mapping_fn` /
callbacks assign each car the learner or a sampled opponent each episode; periodically
add the current policy to the pool. Train on HPC (Ray rollout workers across cores;
`ray[rllib]` in the conda env). Checkpoints → `models/rl_league/`.

## 6. RL-3 — Evaluation triple (outline)
1. **In-sim vs baselines:** drop the trained policy into the validated sim for each
   real race; compare its expected finishing position to time-optimal / position-optimal
   / position+prior picks (reuse Phase-3 harness). Headline number.
2. **Real-scenario counterfactual:** replay the policy in real grids/fields (2022–26);
   positions gained vs what teams actually did (honest about DNFs/noise).
3. **Emergent behaviour:** quantify undercut attempts, SC-reactive pit rate, defensive
   holds; qualitative race traces.

## 7. RL-4 — (optional) serving
Expose the policy as a strategy recommender (FastAPI / frontend hook). Deferred.

---

## 8. Feasibility & risks (honest)
- **Compute is the main risk.** The sim is pure-Python (~30 ms/race); MARL needs
  millions of lap-steps. Ray parallel workers help, but training is heavy (hours–days)
  and may require **vectorising/optimising the sim** (a likely RL-2 sub-task).
- **Deps:** `ray[rllib]` is large; install in the `f1-strategy` conda env on HPC.
- **League self-play is hard to stabilise** (cycling, collapse) — expect iteration; the
  opponent pool + scripted anchors mitigate.
- **HPC constraint:** 2 parallel sbatch jobs; Ray runs single-node across cores.
- **Heterogeneous drivers** handled by driver-conditioned obs; if the shared policy
  underfits fast-vs-slow behaviour, consider a small per-tier policy set (RL-2 fallback).

## 9. File layout (RL-1)
```
src/simulation/multi_car_sim.py        # MODIFY — extract reset()/step(); run() uses them
src/rl/ma_obs.py                       # NEW — pure obs/reward/masking helpers (tested on laptop)
src/rl/multiagent_env.py               # NEW — RLlib MultiAgentEnv on the step-able sim
tests/test_ma_obs.py                   # NEW — pure unit tests
tests/test_multi_car_sim_step.py       # NEW — golden/regression for the step refactor
# RL-2: src/rl/league_train.py, models/rl_league/ ; RL-3: src/analysis/rl_evaluation.py
```

## 10. Implementation note
Umbrella spec across 4 sub-projects; **build RL-1 first** with its own implementation
plan (writing-plans). Re-running Phase 2/3 validation after the step refactor is the
gate for RL-1 — physics must be provably preserved before any training.
