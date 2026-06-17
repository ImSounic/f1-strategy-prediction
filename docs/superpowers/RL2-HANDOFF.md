# RL Workstream — Handoff / Resume Notes

_Last updated: 2026-06-17. Purpose: let a fresh session resume the RL work without re-reading the full chat. Read this + the linked spec/plan and you have everything._

## Where we are right now
- **RL-1: DONE & committed.** `multi_car_sim` is step-able and the RLlib env exists.
- **RL-2 spec: DONE & approved.** Full AlphaStar-style league, staged a/b/c.
- **RL-2a plan: DONE & approved.** Ready to **build inline** (user chose inline). Code NOT written yet.
- **Immediate next action:** write the 3 RL-2a files + 1 test, run the laptop test, hand the user the HPC commands.

## The two-machine workflow (critical)
- **Laptop** (`W:\f1-strategy-prediction`, Windows): Claude edits here. **No pyarrow / no ML stack** → cannot read `.parquet` or run xgboost/torch/ray. Only pure-Python/numpy tests run here.
- **HPC** (`~/f1-strategy-prediction`, conda env `f1-strategy`, Py 3.11): all data/ML/RLlib runs here. Max **2 parallel sbatch**. Ray **2.55** installed (gymnasium pinned 1.2.2).
- **Git:** user commits + pushes manually from laptop, then `git pull` on HPC. Claude never commits.

## RL-2 design (approved)
Spec: [specs/2026-06-16-rl2-league-selfplay-training-design.md](specs/2026-06-16-rl2-league-selfplay-training-design.md)

- **Goal:** SOTA agent via a **full AlphaStar-style league**. User wants max-SOTA for a personal-website showcase; compute/effort cost is explicitly not a concern.
- **Throughput:** Ray parallel rollout workers, **no sim change** (preserve the validated physics — Phase 2/3 Spearman 0.701). Do NOT vectorise the sim unless a later step proves workers too slow.
- **League:** `main`, `main_exploiter`, `league_exploiter` (learning) + frozen `snapshot_*` + `scripted_*` anchors. PFSP matchmaking. A "match" = one 21-car race; pairwise win = car-A finishes ahead of car-B.
- **Curriculum:** single circuit (Bahrain) first → multi-circuit in RL-2c.
- **Staging:** RL-2a (training loop) → RL-2b (league) → RL-2c (scale). Each its own spec→plan→build.

## RL-2a — build this now
Plan (full code in every step): [plans/2026-06-16-rl2a-selfplay-training-loop.md](plans/2026-06-16-rl2a-selfplay-training-loop.md)

Files to create:
- `src/rl/build_env_config.py` — `policy_mapping_fn(agent_id,*a,**k)->"main"` (pure) + `build_env_config(season, circuit_key, config_path)` returning picklable `{circuit, drivers, season}`. Deg-rates baked into `CircuitParams` once via `load_circuit_as_params(...)`, so Ray workers need no xgboost at runtime. **Heavy imports lazy** (inside the function) so the module imports on the laptop.
- `src/rl/train_selfplay.py` — RLlib `PPOConfig`: `.environment("f1_ma", env_config=...)`, `.framework("torch")`, `.multi_agent(policies={"main"}, policy_mapping_fn=...)`, `.env_runners(num_env_runners=K)`, `.training(train_batch_size=4000, gamma=0.99, lr=3e-4)`. Train loop logs `episode_return_mean`, checkpoints to `models/rl_league/selfplay/`.
- `scripts/hpc/train_selfplay.sbatch` — CPU node, ~16 cpus, resumable; knobs `ITERS`/`WORKERS`.
- `tests/test_build_env_config.py` — pure test for `policy_mapping_fn` (laptop). Run: `python tests/test_build_env_config.py` → `1/1 passed`.

**Verification reality:** only the pure test runs on the laptop. Everything else is HPC:
- Smoke: `python -m src.rl.train_selfplay --iters 2 --workers 2 --checkpoint-every 2` → two `episode_return_mean` lines + a checkpoint, no errors.
- Then `sbatch scripts/hpc/train_selfplay.sbatch`.

**RLlib API drift is expected** (new API stack, 2.55). If the smoke errors on a config arg, adjust — likely spots: `.multi_agent(policies={"main"})` form, `.env_runners(num_env_runners=)`, or `config.build()` vs `config.build_algo()`.

**RL-2a gate:** PPO runs to N iters, checkpoints written, `episode_return_mean` **trends up**. Because positions-gained is zero-sum across a uniform-policy field, the real early signal is the agent **eliminating DSQs** (single-compound dry races) so mean return climbs from strongly negative toward ~0; a greedy rollout shows 1-2 stops and ≥2 compounds.

## Env contract (from RL-1, do not break)
`F1MultiAgentEnv(config)` reads `config["circuit"]` (CircuitParams w/ deg_rates), `config["drivers"]` (list[DriverConfig], grid order), `config.get("season",2025)`. Agent ids `car_0..car_{N-1}`. `observation_space` Box(0,1.5,(18,)), `action_space` Discrete(4) = stay / pit-SOFT / pit-MED / pit-HARD. Reward terminal only: `ma_obs.terminal_reward(grid, finish, used_two_compounds)` → positions-gained, or `DSQ_PENALTY=-50.0` if dry race used <2 compounds.

## After RL-2a
- **RL-2b**: `src/rl/league.py` (controller/PFSP via RLlib callbacks: win-rate matrix, snapshot-on-threshold, exploiter resets), exploiter + scripted-anchor policies, league-aware `policy_mapping_fn`, `train_league.py`. Gate: `main` win-rate vs anchors > 50%.
- **RL-2c**: multi-circuit curriculum, longer runs, checkpoint/resume, periodic eval. Use 2 sbatch (train + eval).
- **RL-3**: eval triple — beat Phase-3 baselines in-sim · real-scenario counterfactual · emergent behaviour.
- **RL-4** (optional): serving/recommender.

## Project state (broader)
Step-2 improvement effort: Phases 0–3.6 + Phase 4 (2026 readiness) DONE & validated. See memory `project-phase-state.md` and umbrella spec `specs/2026-06-15-position-validation-and-2026-readiness-design.md`. Known limitation: 2022-25 fuel model mis-corrects lighter 2026 cars (deferred). XGBoost deg model is compound-insensitive → compound realism comes from the regulation-profile multiplier + the temporal prior, not the model.
