# Design: RL-2 — League Self-Play Training (AlphaStar-style)

**Date:** 2026-06-16
**Status:** Approved (pending spec review)

---

## 1. Background & goal

RL-1 delivered `F1MultiAgentEnv` (RLlib `MultiAgentEnv`) on the **validated** step-able
`multi_car_sim`: 21 cars, one shared driver-conditioned policy, Discrete(4) action
(stay / pit-S/M/H) with legality masking, terminal **positions-gained** reward, and a
**disqualification** penalty for the dry-race two-compound rule.

**Goal:** train a SOTA agent via a **full AlphaStar-style league** so it learns pit
strategy (timing + compound) that beats baselines, is realistic enough for a showcase,
and exhibits emergent multi-agent behaviour (undercut/overcut, SC reactivity, defence).

## 2. Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Throughput | **Ray parallel rollout workers** (`num_env_runners`), **no sim change** (validated physics preserved) |
| League | **Full AlphaStar-style**: main agents, main-exploiters, league-exploiters, frozen snapshots, scripted anchors |
| Matchmaking | **PFSP** (win-rate-weighted opponent sampling) |
| "Match" | one 21-car race; pairwise "win" = car-A finishes ahead of car-B |
| Reward | terminal positions-gained (RL-1), DSQ for single-compound dry races |
| Curriculum | start **single circuit** (Bahrain), expand to multi-circuit in RL-2c |
| Framework | RLlib (build on its league-based self-play pattern), staged |
| Compute | HPC; Ray single-node across cores; **2 parallel sbatch jobs** available |

## 3. AlphaStar league → F1

- **Policies (RLlib multi-policy):** `main`, `main_exploiter`, `league_exploiter` (learning)
  + frozen `snapshot_*` + `scripted_*` anchors (non-learning).
- **Per race**, the league controller assigns each of the 21 cars a policy via PFSP +
  role rules; only learning policies' transitions are trained.
- **League controller** (RLlib callbacks): maintains the payoff/win-rate matrix from race
  results, **adds a frozen snapshot** when a learner's league win-rate clears a threshold,
  **resets exploiters** periodically, and drives **PFSP** opponent sampling.
- **Scripted anchors:** the validated fixed/real strategies as permanent opponents — keep
  the league grounded in realistic racecraft and give an absolute yardstick.

## 4. Decomposition (RL-2a → RL-2b → RL-2c; each its own spec→plan→build)

- **RL-2a — Training loop (self-play, single circuit)** *(this spec details it; build first)*
- **RL-2b — Full league** (exploiter populations, snapshot pool, scripted anchors, PFSP, controller)
- **RL-2c — Scale + curriculum** (multi-circuit, longer runs, checkpoints, eval cadence)

Staging de-risks: RL-2a proves the PPO pipeline trains *before* the league machinery.

---

## 5. RL-2a — Training loop (detailed)

### 5.1 What it is
A minimal but real RLlib PPO training run: one shared `main` policy controlling all 21
cars (pure self-play), on **one circuit (Bahrain 2025)**, parallelised with Ray rollout
workers. No league yet. This is the "the pipeline trains and the agent learns the basics"
milestone.

### 5.2 Components
- **Env registration:** `register_env("f1_ma", lambda cfg: F1MultiAgentEnv(cfg))`. The
  `env_config` is fully picklable — `{circuit: CircuitParams (deg_rates pre-baked), drivers:
  [DriverConfig...], season: int}` — so rollout workers need no XGBoost/model at runtime
  (deg rates are baked into `CircuitParams` once, before training).
- **`src/rl/build_env_config.py`** (pure-ish helper): builds the `env_config` from a season
  + circuit key (loads drivers + circuit params once). Unit-testable shape check.
- **`src/rl/train_selfplay.py`:** `PPOConfig` on the new API stack —
  `.environment("f1_ma", env_config=...)`, `.multi_agent(policies={"main"},
  policy_mapping_fn=lambda *a, **k: "main")`, `.env_runners(num_env_runners=K)`,
  `.training(...)` (MLP default). Train loop: `algo.train()` for N iters, log
  `env_runners/episode_return_mean`, checkpoint to `models/rl_league/selfplay/` every M iters.
- **`scripts/hpc/train_selfplay.sbatch`:** CPU node, many cores (Ray workers), `--time` set;
  resumable from checkpoint.

### 5.3 Gate (success for RL-2a)
- Training runs to N iterations on HPC without error; checkpoints written.
- **`episode_return_mean` trends upward** over iterations. (Note: positions-gained is
  zero-sum across a uniform-policy field, so the signal is the agent **eliminating DSQs** and
  learning legal racecraft — mean reward rises toward ~0 as single-compound DSQs vanish.)
- A greedy rollout of the trained policy shows **sane strategy**: ~1–2 stops, ≥2 compounds
  used (no DSQ), pit timing not random.

### 5.4 Tests
- Pure: `build_env_config` returns the expected keys/types; `policy_mapping_fn` returns
  `"main"`. (laptop)
- HPC smoke: 2–3 PPO iterations complete and log a finite `episode_return_mean`; a rollout
  episode runs and the policy's cars use ≥2 compounds.

---

## 6. RL-2b — Full league (outline)
Add `main_exploiter` + `league_exploiter` policies; a **league controller** (RLlib callbacks)
that records pairwise win-rates from each race, samples opponents by **PFSP**, freezes
**snapshots** on win-rate thresholds, and **resets exploiters**; add **scripted anchors**
(validated strategies) as fixed policies. `policy_mapping_fn` becomes league-aware (assigns
each car a role/opponent per episode). Gate: `main` win-rate vs scripted anchors climbs and
holds > 50%.

## 7. RL-2c — Scale + curriculum (outline)
Multi-circuit training (obs already carries circuit constants + driver attrs → one policy
generalises); longer runs; checkpoint/resume; periodic eval snapshots (feeds RL-3). Use the
**2 parallel sbatch** jobs for training + concurrent eval, or two circuit shards.

---

## 8. Compute & ops
- Ray runs single-node across the allocated cores (`num_env_runners` ≈ cores − 2). Policy is
  a small MLP → CPU is fine; no GPU needed.
- Training is long and **resumable from checkpoints**; submit as sbatch (2 parallel max).
- Throughput lever is worker count; the sim is unchanged (no physics risk).

## 9. Risks (honest)
- Full AlphaStar is finicky: exploiter-reset cadence, PFSP weighting, reward scaling, and
  snapshot thresholds all need tuning — expect iteration in RL-2b.
- Pure-Python sim caps per-worker throughput; convergence may take many hours. Staging means
  RL-2a yields a working/improving agent early regardless.
- RLlib new-API-stack churn (2.55) — pin the version; follow the current league example.

## 10. File layout (RL-2a)
```
src/rl/build_env_config.py             # NEW — picklable env_config builder (+ pure test)
src/rl/train_selfplay.py               # NEW — RLlib PPO self-play training loop
scripts/hpc/train_selfplay.sbatch      # NEW — HPC training job (resumable)
tests/test_build_env_config.py         # NEW — pure shape test (laptop)
models/rl_league/selfplay/             # OUTPUT — checkpoints (gitignored if large)
# RL-2b: src/rl/league.py (controller/PFSP), train_league.py ; RL-2c: multi-circuit configs
```

## 11. Implementation note
Umbrella spec across RL-2a/b/c; **build RL-2a first** with its own implementation plan.
RL-2a's gate (PPO trains, reward improves, DSQs eliminated) must hold before adding the league.
