# RL-2a — Self-Play Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Get a real RLlib PPO self-play training run working — one shared `main` policy controlling all 21 cars on the validated env, single circuit (Bahrain), parallelised with Ray rollout workers — proving the training pipeline before the league machinery.

**Architecture:** A picklable `env_config` (deg-rates pre-baked into `CircuitParams`, so workers need no XGBoost) feeds `F1MultiAgentEnv` registered with RLlib; `PPOConfig` trains a single shared policy; an sbatch job runs it on the HPC with checkpoints. No league yet (RL-2b).

**Tech Stack:** Ray RLlib 2.55 (new API stack), PyTorch, the RL-1 env. Only a tiny helper is laptop-testable; training is HPC-verified.

---

## Scope & verification reality
RL-2a is RLlib-heavy → it runs on HPC (Ray + xgboost + data). Laptop checks are limited to
`py_compile` and one pure unit test (`policy_mapping_fn`). The real gate is an **HPC smoke
run** (a few PPO iterations logging `episode_return_mean`). RLlib's new-API-stack arguments
can shift between releases; if the smoke errors on a config arg, that's expected — adjust to
the installed 2.55 surface (the plan notes the likely spots).

## File structure
```
src/rl/build_env_config.py            # NEW — picklable env_config builder + policy_mapping_fn (lazy heavy imports)
src/rl/train_selfplay.py              # NEW — RLlib PPO self-play training loop
scripts/hpc/train_selfplay.sbatch     # NEW — HPC training job (resumable)
tests/test_build_env_config.py        # NEW — pure test for policy_mapping_fn (laptop)
models/rl_league/selfplay/            # OUTPUT — checkpoints
```

---

## Task 1: Env-config builder + policy mapping

**Files:** Create `src/rl/build_env_config.py`; Test `tests/test_build_env_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_env_config.py`:

```python
"""Pure test for the self-play policy mapping. Run: python tests/test_build_env_config.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.build_env_config import policy_mapping_fn


def test_policy_mapping_is_shared_main():
    # Every car maps to the single shared 'main' policy (self-play).
    assert policy_mapping_fn("car_0") == "main"
    assert policy_mapping_fn("car_20", None) == "main"
    assert policy_mapping_fn("car_5", episode=None, worker=None) == "main"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_build_env_config.py`
Expected: `ModuleNotFoundError: No module named 'src.rl.build_env_config'`

- [ ] **Step 3: Write the module**

Create `src/rl/build_env_config.py`:

```python
"""
Env-config + policy mapping for RLlib self-play training (RL-2a)
===============================================================
build_env_config() bakes per-compound degradation into CircuitParams ONCE, so the
returned config is fully picklable and Ray rollout workers run the validated
pure-Python sim with no XGBoost at runtime. Heavy imports are lazy so this module
(and policy_mapping_fn) imports cleanly on any machine.
"""
from __future__ import annotations


def policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """Self-play: every car shares the single 'main' policy."""
    return "main"


def build_env_config(season: int = 2025, circuit_key: str = "bahrain",
                     config_path: str = "configs/config.yaml") -> dict:
    """Build a picklable env_config: {circuit, drivers, season}."""
    import json
    import yaml
    import xgboost as xgb
    from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params

    cfg = yaml.safe_load(open(config_path))
    drivers, _teams, overtaking = load_drivers(f"configs/drivers_{season}.json")
    deg = xgb.XGBRegressor()
    deg.load_model("models/tyre_deg_production.json")
    feature_cols = json.load(open("models/comparison_results.json"))["experiment"]["feature_columns"]
    circuit = load_circuit_as_params(circuit_key, season, cfg, overtaking, deg, feature_cols)
    return {"circuit": circuit, "drivers": drivers, "season": season}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python tests/test_build_env_config.py`
Expected: `1/1 passed` (imports cleanly — heavy deps are lazy).

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: PPO self-play training script

**Files:** Create `src/rl/train_selfplay.py`

- [ ] **Step 1: Write the training script**

Create `src/rl/train_selfplay.py`:

```python
"""
RLlib PPO self-play training (RL-2a)
====================================
One shared 'main' policy controls all cars in F1MultiAgentEnv, single circuit.
Logs episode_return_mean per iteration and checkpoints. No league yet (RL-2b).

Usage:
    python -m src.rl.train_selfplay --iters 50 --workers 6
    python -m src.rl.train_selfplay --iters 2 --workers 2   # smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from src.rl.multiagent_env import F1MultiAgentEnv
from src.rl.build_env_config import build_env_config, policy_mapping_fn


def _episode_return(result: dict):
    """Robustly pull mean episode return across RLlib versions."""
    er = result.get("env_runners", {})
    for key in ("episode_return_mean", "episode_reward_mean"):
        if key in er:
            return er[key]
        if key in result:
            return result[key]
    return None


def main():
    ap = argparse.ArgumentParser(description="RL-2a PPO self-play training")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="models/rl_league/selfplay")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env("f1_ma", lambda cfg: F1MultiAgentEnv(cfg))
    env_config = build_env_config(args.season, args.circuit)

    config = (
        PPOConfig()
        .environment("f1_ma", env_config=env_config)
        .framework("torch")
        .multi_agent(policies={"main"}, policy_mapping_fn=policy_mapping_fn)
        .env_runners(num_env_runners=args.workers)
        .training(train_batch_size=4000, gamma=0.99, lr=3e-4)
    )
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.iters + 1):
        result = algo.train()
        print(f"iter {i:>3} | episode_return_mean {_episode_return(result)}", flush=True)
        if i % args.checkpoint_every == 0 or i == args.iters:
            ckpt = algo.save(str(out))
            print(f"  checkpoint -> {ckpt}", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check (laptop)**

Run: `python -m py_compile src/rl/train_selfplay.py src/rl/build_env_config.py`
Expected: no output (importing needs Ray; this only checks syntax).

- [ ] **Step 3 (HPC): Smoke run — 2 iterations, 2 workers**

Run (in `f1-strategy`):
`python -m src.rl.train_selfplay --iters 2 --workers 2 --checkpoint-every 2`
Expected: two `iter N | episode_return_mean <number>` lines and a `checkpoint -> ...` line,
no errors. If RLlib errors on a config arg (new-API-stack drift in 2.55), adjust the offending
call — likely candidates: `.multi_agent(policies=...)` form, `.env_runners(num_env_runners=...)`,
or `build()` vs `build_algo()` — then re-run. Capture the final form.

- [ ] **Step 4: Commit** (skipped — user commits at end)

---

## Task 3: HPC training job + the RL-2a gate

**Files:** Create `scripts/hpc/train_selfplay.sbatch`

- [ ] **Step 1: Write the sbatch script**

Create `scripts/hpc/train_selfplay.sbatch`:

```bash
#!/usr/bin/env bash
# RL-2a self-play training (single circuit). Submit: sbatch scripts/hpc/train_selfplay.sbatch
#SBATCH --job-name=f1-rl2a
#SBATCH --output=logs/rl2a_%j.out
#SBATCH --error=logs/rl2a_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# #SBATCH --partition=<your_cpu_partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/rl_league/selfplay

# module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy

export CUDA_VISIBLE_DEVICES=""          # CPU-only; policy is a small MLP
ITERS="${ITERS:-200}"
WORKERS="${WORKERS:-14}"                 # ~cpus-per-task - 2 for Ray rollout workers

echo "▸ RL-2a training: iters=$ITERS workers=$WORKERS"
srun python -m src.rl.train_selfplay --iters "$ITERS" --workers "$WORKERS" --checkpoint-every 20
echo "✓ done"
```

- [ ] **Step 2 (HPC): Submit the real run**

Run: `sbatch scripts/hpc/train_selfplay.sbatch` ; watch `squeue --me` and `tail -f logs/rl2a_*.out`.
(Adjust `--cpus-per-task`/partition to your cluster; override knobs with `ITERS=... WORKERS=... sbatch ...`.)

- [ ] **Step 3: Check the RL-2a gate**

Inspect `logs/rl2a_*.out`:
- Training completes the requested iterations and writes checkpoints under `models/rl_league/selfplay/`.
- **`episode_return_mean` trends upward** across iterations. Because positions-gained is
  zero-sum across a uniform-policy field, the signal is the agent **eliminating DSQs**
  (single-compound races), so mean return rises from strongly negative toward ~0.
- If it's flat at a large negative (agents never learn ≥2 compounds), increase `--iters`,
  raise `train_batch_size`, or check the action masking — but the expected trajectory is a
  clear climb in the first ~50–100 iters.

- [ ] **Step 4: Commit** (skipped — user commits at end; checkpoints may be large — gitignore if so)

---

## Self-review

**Spec coverage (RL-2a):**
- Picklable env_config, deg-rates pre-baked, no model in workers → Task 1 `build_env_config`. ✓
- Shared `main` policy self-play, Ray rollout workers, single circuit → Task 2 `train_selfplay`. ✓
- Checkpointing + resumable HPC sbatch → Task 2 (`algo.save`) + Task 3 sbatch. ✓
- Gate (PPO trains, return trends up, DSQs eliminated) → Task 3 Step 3. ✓
- Pure laptop test (policy_mapping_fn) → Task 1. ✓

**Placeholder scan:** none — full code for the helper, training script, and sbatch. The RLlib-API-drift note is an explicit, expected adjustment point, not a placeholder.

**Type consistency:** `policy_mapping_fn` / `build_env_config` signatures match between module, test, and `train_selfplay` imports. Env id `"f1_ma"` consistent across `register_env` and `PPOConfig.environment`. `F1MultiAgentEnv(cfg)` matches the RL-1 constructor (takes a config dict). `env_config` keys `{circuit, drivers, season}` match what `F1MultiAgentEnv.__init__` reads.
