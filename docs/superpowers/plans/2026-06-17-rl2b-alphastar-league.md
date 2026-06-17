# RL-2b — AlphaStar League Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a scaled AlphaStar-style league (2 main + 2 main-exploiter + 2 league-exploiter learners, growing frozen-snapshot pool, scripted anchors, PFSP matchmaking) on top of the RL-2a training loop, so the agent faces non-identical opponents and learns to beat them.

**Architecture:** A pure league core (`league.py`: roles, win-rate matrix, PFSP sampler, field assembly, snapshot/reset predicates) is unit-tested on the laptop. The RLlib layer (`scripted_anchors.py`, `league_callbacks.py`, `train_league.py`) wires it into one multi-policy `Algorithm` and is HPC-verified.

**Tech Stack:** Ray RLlib 2.55 (new API stack), PyTorch, the RL-1 env, the RL-2a env-config.

---

## Build order
Tasks 1–4 are **pure (laptop-tested)** — the league's brain. Tasks 5–8 are **RLlib (HPC-verified)** — the wiring. Prove the brain before the plumbing.

## File structure
```
src/rl/league.py            # roles/config, WinRateMatrix, PFSP, field assembly, predicates (PURE)
tests/test_league.py        # pure tests (laptop)
src/rl/scripted_anchors.py  # deterministic anchor RLModule(s)
src/rl/league_callbacks.py  # RLlib callbacks (thin adapter over league.py)
src/rl/train_league.py      # assemble multi-policy PPOConfig + train loop
scripts/hpc/train_league.sbatch
scripts/hpc/eval_league.sbatch
```

---

## Task 1: Roles, config, WinRateMatrix (pure)
**Files:** Create `src/rl/league.py`; Test `tests/test_league.py`

- [ ] **Step 1: Write failing tests** (matrix records pairwise wins, rate uses prior when unseen)

```python
def test_winrate_records_pairwise():
    m = WinRateMatrix()
    m.record_race([("main_0", 1), ("anchor_x", 2), ("snap_0", 3)])
    assert m.rate("main_0", "anchor_x") == 1.0     # main_0 finished ahead
    assert m.rate("anchor_x", "main_0") == 0.0
    assert m.games("main_0", "snap_0") == 1

def test_winrate_prior_when_unseen():
    m = WinRateMatrix()
    assert m.rate("a", "b") == 0.5                  # no games -> prior
    assert m.rate("a", "b", prior=0.3) == 0.3
```

- [ ] **Step 2: Run, verify fail** — `python tests/test_league.py` → ImportError/fail.

- [ ] **Step 3: Implement** the roles/config + `WinRateMatrix` (see code in Task 1 impl block below).

- [ ] **Step 4: Run, verify pass.**

**Task 1 impl (top of `src/rl/league.py`):**
```python
"""
Pure league core for the AlphaStar-style RL-2b training
=======================================================
No Ray / no torch imports — unit-testable on any machine. The RLlib callbacks
(league_callbacks.py) and trainer (train_league.py) compose these.
"""
from __future__ import annotations

from dataclasses import dataclass

ROLE_MAIN, ROLE_MEXP, ROLE_LEXP = "main", "mexp", "lexp"


@dataclass
class LeagueConfig:
    n_main: int = 2
    n_mexp: int = 2
    n_lexp: int = 2
    focus_share: float = 0.5
    pfsp_p: float = 2.0
    pfsp_eps: float = 0.1
    snapshot_threshold: float = 0.70
    snapshot_every_steps: int = 2_000_000
    exploiter_threshold: float = 0.70
    exploiter_max_steps: int = 4_000_000


def role_of(policy_id: str) -> str:
    """'main_0' -> 'main', 'snap_3' -> 'snap', 'anchor_onestop' -> 'anchor'."""
    return policy_id.split("_")[0]


def learner_ids(cfg: LeagueConfig) -> list:
    return ([f"main_{i}" for i in range(cfg.n_main)]
            + [f"mexp_{i}" for i in range(cfg.n_mexp)]
            + [f"lexp_{i}" for i in range(cfg.n_lexp)])


class WinRateMatrix:
    """Running pairwise win-rates. rate(A,B) = P(an A-car finishes ahead of a B-car)."""

    def __init__(self):
        self._w: dict = {}   # (A,B) -> A-ahead count
        self._g: dict = {}   # (A,B) -> games

    def record_race(self, finishing: list) -> None:
        """finishing: list of (policy_id, finish_position) for every car in the race."""
        n = len(finishing)
        for i in range(n):
            a, pa = finishing[i]
            for j in range(n):
                if i == j:
                    continue
                b, pb = finishing[j]
                if a == b:
                    continue
                self._g[(a, b)] = self._g.get((a, b), 0) + 1
                if pa < pb:
                    self._w[(a, b)] = self._w.get((a, b), 0) + 1

    def rate(self, a: str, b: str, prior: float = 0.5) -> float:
        g = self._g.get((a, b), 0)
        return prior if g == 0 else self._w.get((a, b), 0) / g

    def games(self, a: str, b: str) -> int:
        return self._g.get((a, b), 0)
```

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: PFSP weighting (pure)
**Files:** Modify `src/rl/league.py`; Test `tests/test_league.py`

- [ ] **Step 1: Write failing tests**

```python
def test_pfsp_weights_sum_to_one_and_favor_losses():
    w = pfsp_weights([0.1, 0.9], p=2.0, eps=0.0)   # lose to first opponent
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[1]                              # harder opponent gets more weight

def test_pfsp_eps_gives_coverage():
    w = pfsp_weights([0.0, 1.0], p=2.0, eps=0.2)   # would-be 0 weight opponent
    assert all(x > 0 for x in w)                   # eps keeps everyone reachable

def test_pfsp_uniform_when_equal():
    w = pfsp_weights([0.5, 0.5, 0.5], p=2.0, eps=0.1)
    assert all(abs(x - 1/3) < 1e-9 for x in w)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** (append to `src/rl/league.py`):

```python
def pfsp_weights(rates: list, p: float = 2.0, eps: float = 0.1) -> list:
    """Prioritized fictitious self-play weights. Lower win-rate (harder opponent)
    -> higher weight. Mixed with uniform (eps) so every opponent stays reachable."""
    n = len(rates)
    if n == 0:
        return []
    hard = [max(0.0, 1.0 - r) ** p for r in rates]
    s = sum(hard)
    base = [h / s for h in hard] if s > 0 else [1.0 / n] * n
    return [(1.0 - eps) * b + eps * (1.0 / n) for b in base]
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** (skipped)

---

## Task 3: Role-based opponent pool + field assembly (pure)
**Files:** Modify `src/rl/league.py`; Test `tests/test_league.py`

- [ ] **Step 1: Write failing tests**

```python
def test_opponent_pool_by_role():
    learners = ["main_0", "main_1", "mexp_0", "lexp_0"]
    snaps, anchors = ["snap_0"], ["anchor_onestop"]
    # main-exploiter targets only current mains
    assert set(opponent_pool(ROLE_MEXP, "mexp_0", learners, snaps, anchors)) == {"main_0", "main_1"}
    # main: snapshots + anchors + OTHER mains (not itself)
    assert set(opponent_pool(ROLE_MAIN, "main_0", learners, snaps, anchors)) == {"snap_0", "anchor_onestop", "main_1"}
    # league-exploiter: whole league minus itself
    assert "lexp_0" not in opponent_pool(ROLE_LEXP, "lexp_0", learners, snaps, anchors)

def test_focus_count_and_assemble():
    assert focus_count(20, 0.5) == 10
    field = assemble_field(5, "main_0", n_focus=2, sample_opponent=lambda: "anchor_onestop")
    assert field == ["main_0", "main_0", "anchor_onestop", "anchor_onestop", "anchor_onestop"]
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** (append):

```python
def opponent_pool(role: str, focus_id: str, learners: list,
                  snapshots: list, anchors: list) -> list:
    """Candidate opponents for a focus policy, by AlphaStar role."""
    if role == ROLE_MEXP:                                   # main-exploiter: hunt mains
        return [x for x in learners if role_of(x) == ROLE_MAIN]
    if role == ROLE_MAIN:                                   # main: pool + other mains
        other_mains = [x for x in learners if role_of(x) == ROLE_MAIN and x != focus_id]
        return list(snapshots) + list(anchors) + other_mains
    return list(snapshots) + list(anchors) + [x for x in learners if x != focus_id]  # lexp


def focus_count(n_cars: int, focus_share: float) -> int:
    return max(1, min(n_cars, round(n_cars * focus_share)))


def assemble_field(n_cars: int, focus_id: str, n_focus: int, sample_opponent) -> list:
    """Grid of policy ids: n_focus cars are the focus policy, the rest sampled opponents."""
    field = [focus_id] * min(n_focus, n_cars)
    while len(field) < n_cars:
        field.append(sample_opponent())
    return field
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** (skipped)

---

## Task 4: Snapshot / reset predicates (pure)
**Files:** Modify `src/rl/league.py`; Test `tests/test_league.py`

- [ ] **Step 1: Write failing tests**

```python
def test_league_winrate_average():
    m = WinRateMatrix()
    m.record_race([("main_0", 1), ("snap_0", 2)])   # beats snap_0
    m.record_race([("main_0", 2), ("anchor_x", 1)]) # loses to anchor_x
    wr = league_winrate(m, "main_0", ["snap_0", "anchor_x"])
    assert abs(wr - 0.5) < 1e-9

def test_snapshot_and_reset_triggers():
    m = WinRateMatrix()
    for _ in range(10):
        m.record_race([("main_0", 1), ("snap_0", 2), ("anchor_x", 3)])
    assert should_snapshot(m, "main_0", ["snap_0", "anchor_x"], steps_since=0,
                           threshold=0.7, every_steps=10**9) is True
    assert should_reset_exploiter(m, "mexp_0", ["main_0"], steps_alive=0,
                                  threshold=0.7, max_steps=10**9) is False  # never beat main_0
    assert should_reset_exploiter(m, "mexp_0", ["main_0"], steps_alive=10**10,
                                  threshold=0.7, max_steps=10**9) is True   # aged out
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** (append):

```python
def league_winrate(matrix: WinRateMatrix, agent: str, opponents: list,
                   prior: float = 0.5) -> float:
    if not opponents:
        return prior
    return sum(matrix.rate(agent, o, prior) for o in opponents) / len(opponents)


def should_snapshot(matrix: WinRateMatrix, main_id: str, league_opponents: list,
                    steps_since: int, threshold: float, every_steps: int) -> bool:
    return (league_winrate(matrix, main_id, league_opponents) >= threshold
            or steps_since >= every_steps)


def should_reset_exploiter(matrix: WinRateMatrix, exp_id: str, targets: list,
                           steps_alive: int, threshold: float, max_steps: int) -> bool:
    return (league_winrate(matrix, exp_id, targets) >= threshold
            or steps_alive >= max_steps)
```

- [ ] **Step 4: Run, verify pass — all pure league tests green.**
- [ ] **Step 5: Commit** (skipped)

---

## Task 5: Scripted anchor RLModule (HPC)
**Files:** Create `src/rl/scripted_anchors.py`

A frozen RLModule that ignores learning and emits a deterministic action from a fixed
strategy plan. The obs encodes lap fraction (index 0) and tyre age (index 1); a plan is a
list of `(pit_lap_fraction, action_int)` — pit to a compound at given race fractions, else stay (0).

- [ ] **Step 1: Implement**

```python
"""
Scripted anchor policies for the RL-2b league
=============================================
Deterministic, frozen RLModules that play a fixed pit plan. They race in-env as
opponents and form the absolute yardstick for the RL-2b gate. New API stack.
"""
from __future__ import annotations

import torch
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI

# action ints match ma_obs: 0 stay, 1 pit-SOFT, 2 pit-MED, 3 pit-HARD
ANCHOR_PLANS = {
    "anchor_onestop": [(0.55, 2)],                 # one stop ~55% distance to MEDIUM
    "anchor_twostop": [(0.35, 2), (0.70, 3)],      # two stops: MED then HARD
}


class ScriptedAnchorModule(TorchRLModule, InferenceOnlyAPI):
    """Emits a pit action when lap-fraction first crosses a planned pit point."""

    def __init__(self, *args, plan=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = plan or []

    def _action(self, obs_row):
        lap_frac = float(obs_row[0])
        act = 0
        for frac, a in self.plan:
            # fire the pit on the first step at/after the planned fraction within a small window
            if frac <= lap_frac < frac + 0.03:
                act = a
        return act

    def _forward_inference(self, batch, **kwargs):
        obs = batch["obs"]
        acts = torch.tensor([self._action(row) for row in obs], dtype=torch.int64)
        return {"actions": acts}

    _forward_exploration = _forward_inference
    _forward_train = _forward_inference
```

- [ ] **Step 2: Syntax check (laptop)** — `python -m py_compile src/rl/scripted_anchors.py`.
- [ ] **Step 3: Verified on HPC** via the Task 7 smoke (anchors must act without error). **API-drift note:** the exact `TorchRLModule` forward signature / required APIs may differ in 2.55 — if the smoke errors here, adjust the module to the installed `rllib.core.rl_module` surface (most likely the `_forward_*` return key or the constructor kwargs).
- [ ] **Step 4: Commit** (skipped)

---

## Task 6: League callbacks (HPC)
**Files:** Create `src/rl/league_callbacks.py`

Thin adapter: records finishing order into a `WinRateMatrix` on episode end, and (later) the
trainer queries the pure predicates to add snapshots / reset exploiters.

- [ ] **Step 1: Implement**

```python
"""
RLlib callbacks for the RL-2b league (thin adapter over league.py)
=================================================================
on_episode_end records the race's finishing order into a shared WinRateMatrix,
keyed by each car's policy id (resolved via the episode's agent->module mapping).
Snapshot/reset decisions live in train_league.py using league.py predicates.
"""
from __future__ import annotations

from ray.rllib.callbacks.callbacks import RLlibCallback

from src.rl.league import WinRateMatrix


class LeagueCallbacks(RLlibCallback):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        algorithm._winrates = WinRateMatrix()

    def on_episode_end(self, *, episode, env_runner=None, metrics_logger=None,
                       env=None, **kwargs):
        # Build (policy_id, finish_position) per agent from the terminal info/rewards.
        # finish position = rank by terminal reward is NOT valid (DSQ); use env state.
        order = []
        try:
            base = env.unwrapped if env is not None else None
            sim = getattr(base, "sim", None)
            if sim is None:
                return
            for i, agent_id in enumerate(base.agents):
                pid = episode.module_for(agent_id) if hasattr(episode, "module_for") else "main_0"
                order.append((pid, int(sim.positions[i])))
        except Exception:  # noqa: BLE001  — never crash training on bookkeeping
            return
        # Stash on the episode; the env_runner aggregates to the driver via metrics.
        if metrics_logger is not None:
            metrics_logger.log_value("league/race_recorded", 1.0, reduce="sum")
        # Record locally if the matrix is reachable (single-runner / local case).
        wr = getattr(env_runner, "_winrates", None)
        if wr is not None:
            wr.record_race(order)
```

- [ ] **Step 2: Syntax check (laptop)** — `python -m py_compile src/rl/league_callbacks.py`.
- [ ] **Step 3: Verified on HPC** (Task 7 smoke). **API-drift note:** callback hook names/signatures and how to resolve an agent's policy id (`episode.module_for`) vary across 2.55 point releases; the smoke will surface the exact signature. Win-rate aggregation across remote runners may need a metrics round-trip — for the smoke (few workers) local recording is enough; full aggregation is a tuning item.
- [ ] **Step 4: Commit** (skipped)

---

## Task 7: League trainer (HPC)
**Files:** Create `src/rl/train_league.py`

- [ ] **Step 1: Implement**

```python
"""
RL-2b league trainer
====================
One PPO Algorithm with multiple learning policies (main/mexp/lexp) + frozen
scripted anchors; league-aware policy_mapping_fn assigns each car a policy per
episode via PFSP; LeagueCallbacks records win-rates; snapshots/resets applied
between iterations using league.py predicates.

Usage:
    python -m src.rl.train_league --iters 2 --workers 2     # smoke
    python -m src.rl.train_league --iters 300 --workers 14
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from src.rl.multiagent_env import F1MultiAgentEnv
from src.rl.build_env_config import build_env_config
from src.rl.league import (
    LeagueConfig, learner_ids, role_of, opponent_pool, focus_count,
    assemble_field, pfsp_weights, ROLE_MAIN, ROLE_MEXP, ROLE_LEXP,
)
from src.rl.league_callbacks import LeagueCallbacks
from src.rl.scripted_anchors import ScriptedAnchorModule, ANCHOR_PLANS


def build_assignment(cfg, learners, anchors, n_cars, matrix, rng):
    """Pick a focus learner, assemble the per-car policy grid via PFSP."""
    focus = rng.choice(learners)
    role = role_of(focus)
    pool = opponent_pool(role, focus, learners, snapshots=[], anchors=anchors)
    if not pool:
        pool = anchors or learners

    def sample_opponent():
        rates = [matrix.rate(focus, o) for o in pool]
        w = pfsp_weights(rates, p=cfg.pfsp_p, eps=cfg.pfsp_eps)
        return rng.choices(pool, weights=w, k=1)[0]

    n_focus = focus_count(n_cars, cfg.focus_share)
    return assemble_field(n_cars, focus, n_focus, sample_opponent)


def main():
    ap = argparse.ArgumentParser(description="RL-2b league trainer")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--out", type=str, default="models/rl_league/league")
    args = ap.parse_args()

    cfg = LeagueConfig()
    learners = learner_ids(cfg)
    anchors = list(ANCHOR_PLANS.keys())
    rng = random.Random(0)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    env_config = build_env_config(args.season, args.circuit)
    register_env("f1_ma", lambda c: F1MultiAgentEnv(c))
    n_cars = len(env_config["drivers"])

    # Shared assignment state, regenerated per episode by the mapping fn.
    state = {"field": build_assignment(cfg, learners, anchors, n_cars,
                                       _MatrixStub(), rng)}

    def policy_mapping_fn(agent_id, episode=None, **kw):
        i = int(str(agent_id).split("_")[1])
        field = state["field"]
        return field[i % len(field)]

    # All policies: learners (trained) + anchors (frozen scripted modules).
    module_specs = {pid: RLModuleSpec() for pid in learners}
    for a in anchors:
        module_specs[a] = RLModuleSpec(module_class=ScriptedAnchorModule,
                                       model_config={"plan": ANCHOR_PLANS[a]})

    config = (
        PPOConfig()
        .environment("f1_ma", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=set(learners) | set(anchors),
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=learners,           # anchors frozen
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs))
        .env_runners(num_env_runners=args.workers)
        .callbacks(LeagueCallbacks)
        .training(train_batch_size=4000, num_epochs=args.num_epochs, gamma=0.99, lr=3e-4)
    )
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.iters + 1):
        # New matchmaking each iter (simple: refresh the focus/field).
        state["field"] = build_assignment(cfg, learners, anchors, n_cars,
                                           getattr(algo, "_winrates", _MatrixStub()), rng)
        result = algo.train()
        er = result.get("env_runners", {})
        print(f"iter {i:>3} | return {er.get('episode_return_mean')} "
              f"| field_focus {state['field'][0]}", flush=True)
        if i % args.checkpoint_every == 0 or i == args.iters:
            print(f"  checkpoint -> {algo.save(str(out))}", flush=True)

    algo.stop()
    ray.shutdown()


class _MatrixStub:
    """Used before any race is recorded: every rate is the 0.5 prior."""
    def rate(self, a, b, prior=0.5):
        return prior


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check (laptop)** — `python -m py_compile src/rl/train_league.py`.
- [ ] **Step 3: HPC smoke** — `python -m src.rl.train_league --iters 2 --workers 2 --checkpoint-every 2`. Expect 2 `iter N | return ... | field_focus ...` lines + a checkpoint, no errors. **This is where the RLlib new-API multi-policy + custom-RLModule wiring gets validated** — expect to iterate on: `MultiRLModuleSpec`/`RLModuleSpec` argument names, the scripted module forward signature (Task 5), the callback hook signatures (Task 6), and `policies_to_train` handling of frozen modules. Paste any traceback.
- [ ] **Step 4: Commit** (skipped)

---

## Task 8: HPC jobs + RL-2b gate
**Files:** Create `scripts/hpc/train_league.sbatch`, `scripts/hpc/eval_league.sbatch`

- [ ] **Step 1: Training sbatch**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=f1-rl2b
#SBATCH --output=logs/rl2b_%j.out
#SBATCH --error=logs/rl2b_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# #SBATCH --partition=<your_cpu_partition>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/rl_league/league
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy
export CUDA_VISIBLE_DEVICES=""
ITERS="${ITERS:-300}"; WORKERS="${WORKERS:-14}"
echo "▸ RL-2b league: iters=$ITERS workers=$WORKERS (6 learners — expect ~6x RL-2a/iter)"
srun python -m src.rl.train_league --iters "$ITERS" --workers "$WORKERS" --checkpoint-every 20
echo "✓ done"
```

- [ ] **Step 2: Eval sbatch (2nd parallel job)** — placeholder that loads the latest checkpoint
  and races each `main` agent vs the scripted anchors, printing win-rate. (Full eval is RL-3;
  this is the live monitor.)

```bash
#!/usr/bin/env bash
#SBATCH --job-name=f1-rl2b-eval
#SBATCH --output=logs/rl2b_eval_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy
export CUDA_VISIBLE_DEVICES=""
python -m src.rl.train_league --iters 0 --workers 1   # build-only smoke; full eval lands in RL-3
echo "✓ eval stub ok"
```

- [ ] **Step 3: Submit + watch** — `sbatch scripts/hpc/train_league.sbatch`; `tail -f logs/rl2b_*.out`.

- [ ] **Step 4: Check RL-2b gate** in the log:
  - `episode_return_mean` is **non-zero and positive-trending for focus=main** races (the
    learner is now beating weaker anchors/snapshots — unlike RL-2a's flat 0).
  - Win-rate of `main` vs anchors (from the eval job / `_winrates`) **climbs and holds > 50%**.
  - Checkpoints written; training stable (no NaNs / collapse).
  - If `episode_return_mean` stays ~0, check the field actually contains anchors (focus_share,
    `policy_mapping_fn`), since identical-policy fields are still zero-sum.

- [ ] **Step 5: Commit** (skipped — user commits; gitignore `models/rl_league/`)

---

## Self-review

**Spec coverage:** populations/IDs → Task 1 (`learner_ids`,`role_of`); win-rate matrix → Task 1; PFSP → Task 2; role pool + mixed field/focus_share → Task 3; snapshot/reset predicates → Task 4; scripted anchors → Task 5; callbacks (record results) → Task 6; one-Algorithm multi-policy + league-aware mapping + PFSP draw → Task 7; sbatch + 2nd eval job + gate → Task 8. Reward unchanged (RL-1) — no task needed. ✓

**Placeholder scan:** none — full code in every task. The eval sbatch is explicitly a build-only stub with full eval deferred to RL-3 (stated, not a hidden gap). RLlib-API-drift notes in Tasks 5–7 are explicit expected-adjustment points, consistent with how RL-2a actually went.

**Type consistency:** `WinRateMatrix.rate(a,b,prior)` / `record_race(list[(id,pos)])` / `games` used identically across Tasks 1,4,7. `role_of`/`learner_ids`/`opponent_pool`/`pfsp_weights`/`focus_count`/`assemble_field`/`league_winrate`/`should_snapshot`/`should_reset_exploiter` signatures match between `league.py`, the tests, and `train_league.py`. Action ints in `scripted_anchors` (0/1/2/3) match `ma_obs._ACTION_COMPOUND`. Env id `"f1_ma"` and `build_env_config` keys consistent with RL-2a.

**Snapshot/reset wiring caveat (known gap to close on HPC):** Tasks 1–4 provide the predicates and Task 6 records win-rates, but Task 7's loop does not yet *call* `algo.add_policy` for snapshots / reinit exploiters — that runtime step is deliberately left for the first HPC iteration once the multi-policy build is confirmed stable (it's the highest-API-risk piece). RL-2b's first milestone is: multi-policy league trains + win-rates recorded + main beats anchors; live snapshot-adding/reset is the immediate follow-on within RL-2b.
