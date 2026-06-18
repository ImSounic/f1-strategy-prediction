# RL-3a — Eval Harness + Beat-Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Rigorously measure the trained RL `main` agent against the scripted anchors and the project's MC/Phase-3 recommended strategy, head-to-head in the validated sim, reporting win-rate and mean finishing position with 95% confidence intervals.

**Architecture:** Pluggable per-car *controllers* (RL / scripted / MC-plan) drive the validated `multi_car_sim` directly (no RLlib at eval). A neutral identical-driver field isolates *strategy* quality. Pure stats + controller/plan logic are laptop-tested; the model forward + race runs are HPC-verified.

**Tech Stack:** the RL-1 `ma_obs` helpers, `multi_car_sim`, the trained checkpoint, numpy, torch (HPC only).

---

## Build order
Tasks 1–3 are **pure (laptop-tested)**. Tasks 4–8 are **HPC-verified** (need torch + the trained model + the sim with a real circuit).

## File structure
```
src/rl/eval/__init__.py
src/rl/eval/metrics.py          # PURE: win_rate, mean_finish, bootstrap_ci
src/rl/eval/plans.py            # PURE: parse MC default_plan + anchor plans -> (start_compound, plan)
src/rl/eval/controllers.py      # ScriptedController (pure) + RLController (torch, HPC)
src/rl/eval/obs_builder.py      # car_obs(sim, circuit, profile, drivers, i) — shared with the env
src/rl/eval/model_loader.py     # load the trained main module for inference (HPC)
src/rl/eval/race_runner.py      # run_race(...) over multi_car_sim with controllers (HPC)
src/rl/eval/run_beat_baselines.py  # orchestrate N seeds -> JSON + figure (HPC)
scripts/hpc/eval_rl.sbatch
tests/test_eval_metrics.py
tests/test_eval_plans.py
tests/test_eval_controllers.py
results/rl_eval/beat_baselines.json   # OUTPUT (+ figures/)
```

---

## Task 1: Metrics (pure)
**Files:** Create `src/rl/eval/__init__.py` (empty), `src/rl/eval/metrics.py`; Test `tests/test_eval_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
"""Pure tests for eval metrics. Run: python tests/test_eval_metrics.py"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rl.eval.metrics import win_rate, mean_finish, bootstrap_ci


def test_win_rate_paired():
    # a beats b in 3 of 4 races (lower finish position is better)
    assert win_rate([1, 2, 1, 5], [3, 4, 2, 1]) == 0.75


def test_win_rate_ties_excluded():
    # ties (equal finish) are not wins; counted in denominator
    assert win_rate([2, 2], [2, 1]) == 0.0


def test_mean_finish():
    assert mean_finish([2, 4, 6]) == 4.0


def test_bootstrap_ci_deterministic_and_brackets_mean():
    lo, hi = bootstrap_ci([2.0, 4.0, 6.0, 8.0], n=500, seed=0)
    lo2, hi2 = bootstrap_ci([2.0, 4.0, 6.0, 8.0], n=500, seed=0)
    assert (lo, hi) == (lo2, hi2)          # deterministic with seed
    assert lo <= 5.0 <= hi                 # brackets the sample mean (5.0)
```

- [ ] **Step 2: Run, verify fail** — `python tests/test_eval_metrics.py` → ImportError.

- [ ] **Step 3: Implement** `src/rl/eval/metrics.py`:

```python
"""Pure evaluation statistics (win-rate, mean finish, bootstrap CIs)."""
from __future__ import annotations

import numpy as np


def win_rate(a_finishes, b_finishes) -> float:
    """Fraction of paired races where a finished ahead of b (lower pos = better).
    Ties are not wins. Lists must be equal length and non-empty."""
    a, b = list(a_finishes), list(b_finishes)
    assert len(a) == len(b) and a, "need equal-length non-empty paired finishes"
    wins = sum(1 for x, y in zip(a, b) if x < y)
    return wins / len(a)


def mean_finish(finishes) -> float:
    vals = list(finishes)
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean. Deterministic given seed."""
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=vals.size, replace=True).mean()
                      for _ in range(n)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)
```

- [ ] **Step 4: Run, verify pass** — `python tests/test_eval_metrics.py` → `4/4 passed`.
- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: Plan parsing (pure)
**Files:** Create `src/rl/eval/plans.py`; Test `tests/test_eval_plans.py`

A "plan" is `(start_compound, [(lap_fraction, action_int), ...])` where action_int matches
`ma_obs._ACTION_COMPOUND` (1 SOFT, 2 MEDIUM, 3 HARD). The MC recommendation in
`results/scenarios_<circuit>_<season>.json` `default_plan` encodes stints in its `name`
(e.g. `"2-stop HARD→MEDIUM→SOFT (18/18/21)"`) + `compound_sequence`.

- [ ] **Step 1: Write failing tests**

```python
"""Pure tests for eval plan parsing. Run: python tests/test_eval_plans.py"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rl.eval.plans import parse_mc_plan, anchor_plan, ACTION_FOR


def test_action_for():
    assert (ACTION_FOR["SOFT"], ACTION_FOR["MEDIUM"], ACTION_FOR["HARD"]) == (1, 2, 3)


def test_parse_mc_two_stop():
    start, plan = parse_mc_plan("2-stop HARD→MEDIUM→SOFT (18/18/21)",
                                "HARD → MEDIUM → SOFT", total_laps=57)
    assert start == "HARD"
    # pits after lap 18 (->MEDIUM) and lap 36 (->SOFT)
    assert plan[0][1] == ACTION_FOR["MEDIUM"] and abs(plan[0][0] - 18/57) < 1e-9
    assert plan[1][1] == ACTION_FOR["SOFT"] and abs(plan[1][0] - 36/57) < 1e-9
    assert len(plan) == 2


def test_parse_mc_one_stop():
    start, plan = parse_mc_plan("1-stop MEDIUM→HARD (28/29)", "MEDIUM → HARD", total_laps=57)
    assert start == "MEDIUM" and len(plan) == 1 and plan[0][1] == ACTION_FOR["HARD"]


def test_anchor_plans_start_medium_and_switch():
    s1, p1 = anchor_plan("onestop")
    s2, p2 = anchor_plan("twostop")
    assert s1 == "MEDIUM" and p1 == [(0.55, ACTION_FOR["HARD"])]
    assert s2 == "MEDIUM" and len(p2) == 2
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `src/rl/eval/plans.py`:

```python
"""Pure helpers to express baseline strategies as controller plans.

A plan is (start_compound, [(lap_fraction, action_int), ...]). action_int matches
ma_obs: 1=SOFT, 2=MEDIUM, 3=HARD.
"""
from __future__ import annotations

import json
import re

ACTION_FOR = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}

# Cars start on MEDIUM (training default); a 1-stop must switch compound.
_ANCHORS = {
    "onestop": ("MEDIUM", [(0.55, ACTION_FOR["HARD"])]),
    "twostop": ("MEDIUM", [(0.30, ACTION_FOR["SOFT"]), (0.65, ACTION_FOR["HARD"])]),
}


def anchor_plan(kind: str):
    return _ANCHORS[kind]


def parse_mc_plan(name: str, compound_sequence: str, total_laps: int):
    """Parse a scenarios default_plan into (start_compound, plan).

    name carries stint lengths in parens, e.g. '... (18/18/21)';
    compound_sequence is 'A -> B -> C' (arrow may be unicode →).
    """
    compounds = [c.strip().upper()
                 for c in compound_sequence.replace("→", "->").split("->")]
    m = re.search(r"\(([\d/]+)\)", name)
    laps = [int(x) for x in m.group(1).split("/")] if m else []
    plan = []
    cum = 0
    for k in range(len(laps) - 1):                  # final stint has no pit
        cum += laps[k]
        plan.append((cum / total_laps, ACTION_FOR[compounds[k + 1]]))
    return compounds[0], plan


def load_mc_plan(scenarios_path: str, total_laps: int):
    """Load and parse the MC default_plan from a scenarios_<circuit>_<season>.json."""
    d = json.load(open(scenarios_path))
    dp = d["default_plan"]
    return parse_mc_plan(dp["name"], dp["compound_sequence"], total_laps)
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** (skipped)

---

## Task 3: Scripted controller (pure)
**Files:** Create `src/rl/eval/controllers.py` (ScriptedController only for now); Test `tests/test_eval_controllers.py`

- [ ] **Step 1: Write failing tests**

```python
"""Pure tests for the scripted controller. Run: python tests/test_eval_controllers.py"""
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rl.eval.controllers import ScriptedController


def _obs(lap_frac):
    v = np.zeros(18, dtype=np.float32); v[0] = lap_frac; return v


def test_scripted_fires_in_window_else_stays():
    c = ScriptedController(start_compound="MEDIUM", plan=[(0.55, 3)])
    assert c.decide(_obs(0.10)) == 0          # before pit point -> stay
    assert c.decide(_obs(0.56)) == 3          # inside window -> pit to HARD
    assert c.decide(_obs(0.90)) == 0          # after window -> stay
    assert c.start_compound == "MEDIUM"
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `src/rl/eval/controllers.py`:

```python
"""Per-car controllers for evaluation races.

A controller maps a (normalised) observation -> Discrete(4) action int
(0 stay, 1 pit-SOFT, 2 pit-MED, 3 pit-HARD) and declares a start_compound.
ScriptedController is pure; RLController wraps a trained module (added in Task 6).
"""
from __future__ import annotations

_PIT_WINDOW = 0.03   # ~1.7 laps at 57 laps; the legality mask blocks a double-pit


class ScriptedController:
    def __init__(self, start_compound: str, plan):
        self.start_compound = start_compound
        self.plan = list(plan)            # [(lap_fraction, action_int), ...]

    def decide(self, obs) -> int:
        lap_frac = float(obs[0])
        for frac, action in self.plan:
            if frac <= lap_frac < frac + _PIT_WINDOW:
                return int(action)
        return 0
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** (skipped)

---

## Task 4: Shared observation builder (HPC-verified)
**Files:** Create `src/rl/eval/obs_builder.py`; Modify `src/rl/multiagent_env.py` (`_obs_for` delegates to it)

The RLController MUST see the *exact* observation the policy trained on. Extract the env's
`_obs_for` body into a free function and have both the env and the race runner use it.

- [ ] **Step 1: Implement** `src/rl/eval/obs_builder.py` (mirrors `F1MultiAgentEnv._obs_for` verbatim):

```python
"""Build the per-car observation from sim state — shared by the training env and
the eval race runner so the policy sees identical inputs in both."""
from __future__ import annotations

import numpy as np

from src.rl.ma_obs import build_obs

_COMPOUND_IDX = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}


def car_obs(sim, circuit, profile, drivers, i: int) -> np.ndarray:
    car = sim.cars[i]
    driver = drivers[i]
    n_cars = len(drivers)
    gaps = sim._compute_gaps(sim.cars, sim.positions)
    order = list(np.argsort(sim.positions))
    pos_idx = order.index(i)
    gap_behind = gaps[order[pos_idx + 1]] if pos_idx + 1 < n_cars else 999.0
    state = dict(
        lap=sim.lap, total_laps=circuit.total_laps,
        tyre_age=car.tyre_age, compound_idx=_COMPOUND_IDX.get(car.tyre_compound, 1),
        cumulative_deg=0.0, position=int(sim.positions[i]), n_cars=n_cars,
        gap_ahead=float(gaps[i]), gap_behind=float(gap_behind),
        fuel_frac=max(0.0, 1.0 - sim.burn_rate * sim.lap / max(profile.start_fuel_kg, 1)),
        sc_active=int(sim.sc_active or sim.vsc_active),
        stops_done=car.stops_done, max_stops=3, compounds_used=len(car.compounds_used),
        laps_since_sc=0,
        driver_pace=min(driver.pace_delta, 2.0) / 2.0,
        driver_overtaking=driver.overtaking, driver_tyre=driver.tyre_management,
        pit_loss=circuit.pit_loss_seconds, sc_prob=circuit.sc_prob_per_race,
        overtaking_difficulty=circuit.overtaking_difficulty,
    )
    return build_obs(state)
```

- [ ] **Step 2: Modify** `src/rl/multiagent_env.py` — replace the `_obs_for` body with a delegation (keeps behaviour identical). Change the method to:

```python
    def _obs_for(self, i: int) -> np.ndarray:
        from src.rl.eval.obs_builder import car_obs
        return car_obs(self.sim, self.circuit, self.profile, self.drivers, i)
```

(Delete the old inline body; the `_COMPOUND_IDX` constant at module top in multiagent_env.py
may remain unused — leave it, harmless.)

- [ ] **Step 3: Syntax check (laptop)** — `python -m py_compile src/rl/eval/obs_builder.py src/rl/multiagent_env.py`.
- [ ] **Step 4: HPC verify** — covered by the Task 8 smoke (env still trains-compatible; obs identical). If a 2-iter `train_league` smoke still runs, the extraction preserved behaviour.
- [ ] **Step 5: Commit** (skipped)

---

## Task 5: Model loader (HPC)
**Files:** Create `src/rl/eval/model_loader.py`

- [ ] **Step 1: Implement**

```python
"""Load the trained main module from a league checkpoint for inference."""
from __future__ import annotations

from pathlib import Path


def load_main_module(checkpoint: str, module_id: str = "main_1"):
    """Return the trained RLModule for module_id (default the stronger main agent).
    Uses the validated Algorithm.from_checkpoint, then extracts one module."""
    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(str(Path(checkpoint).resolve()))
    module = algo.get_module(module_id)
    return module, algo     # keep algo alive so the module isn't GC'd; caller stops it
```

- [ ] **Step 2: Syntax check** — `python -m py_compile src/rl/eval/model_loader.py`.
- [ ] **Step 3: HPC verify** — exercised in Task 8 smoke. **API-drift note:** if `get_module`
  needs a different accessor on 2.55, adjust (e.g. `algo.get_module(module_id)` vs
  `algo.learner_group...`); the from_checkpoint half is already validated.
- [ ] **Step 4: Commit** (skipped)

---

## Task 6: RL controller (HPC)
**Files:** Modify `src/rl/eval/controllers.py` (add RLController)

- [ ] **Step 1: Implement** — append to `src/rl/eval/controllers.py`:

```python
class RLController:
    """Greedy controller wrapping a trained RLModule. start_compound matches training
    (cars started on MEDIUM during training)."""

    def __init__(self, module, start_compound: str = "MEDIUM"):
        self.module = module
        self.start_compound = start_compound

    def decide(self, obs) -> int:
        import torch
        with torch.no_grad():
            batch = {"obs": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
            out = self.module.forward_inference(batch)
            if "actions" in out:
                return int(out["actions"][0])
            logits = out["action_dist_inputs"]            # Discrete -> argmax = greedy
            return int(torch.argmax(logits[0]).item())
```

- [ ] **Step 2: Syntax check** — `python -m py_compile src/rl/eval/controllers.py` (laptop; torch import is lazy).
- [ ] **Step 3: HPC verify** — Task 8 smoke. **API-drift note:** the forward output key
  (`action_dist_inputs` vs `actions`) is handled both ways; if the module exposes neither,
  print `out.keys()` and adjust.
- [ ] **Step 4: Commit** (skipped)

---

## Task 7: Race runner (HPC)
**Files:** Create `src/rl/eval/race_runner.py`

- [ ] **Step 1: Implement**

```python
"""Run one evaluation race over the validated multi_car_sim with pluggable controllers."""
from __future__ import annotations

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy
from src.rl.ma_obs import action_to_compound, legal_action_mask
from src.rl.eval.obs_builder import car_obs


def run_race(circuit, drivers, controllers, profile, seed: int):
    """controllers: list aligned with drivers; each has .start_compound and .decide(obs).
    Returns finishing position per car index (lower = better)."""
    n = len(drivers)
    # single-stint strategies (no auto-pit); pits are driven entirely by controllers.
    strategies = [Strategy(stints=[(controllers[i].start_compound, circuit.total_laps)],
                           name=f"car_{i}") for i in range(n)]
    default_strat = Strategy(stints=[("MEDIUM", circuit.total_laps)], name="x")
    sim = MultiCarRaceSim(circuit, drivers, strategies, 0, default_strat,
                          greedy_sc=False, profile=profile)
    sim.reset(seed=seed)
    while not sim.done:
        pit_override = {}
        for i in range(n):
            obs = car_obs(sim, circuit, profile, drivers, i)
            act = int(controllers[i].decide(obs))
            mask = legal_action_mask(sim.cars[i].stops_done, sim.cars[i].tyre_age,
                                     sim.lap + 1, circuit.total_laps)
            pit_override[i] = action_to_compound(act) if mask[act] else None
        sim.step(pit_override)
    return [int(p) for p in sim.positions]
```

- [ ] **Step 2: Syntax check** — `python -m py_compile src/rl/eval/race_runner.py`.
- [ ] **Step 3: HPC verify** — Task 8 smoke (a race completes and returns 21 finite positions).
- [ ] **Step 4: Commit** (skipped)

---

## Task 8: Orchestrator + sbatch + the RL-3a gate (HPC)
**Files:** Create `src/rl/eval/run_beat_baselines.py`, `scripts/hpc/eval_rl.sbatch`

Neutral identical-driver field isolates *strategy*: all 21 cars use one representative driver
profile, differing only by controller. Field = 1 RL car (rotating slot) + 20 baselines split
across {anchor onestop, anchor twostop, mc}. Aggregate over N seeds with bootstrap CIs.

- [ ] **Step 1: Implement** `src/rl/eval/run_beat_baselines.py`:

```python
"""RL-3a: head-to-head RL main vs anchors + MC/Phase-3 plan, with bootstrap CIs.

Usage:
    python -m src.rl.eval.run_beat_baselines --races 8 --circuit bahrain   # smoke
    python -m src.rl.eval.run_beat_baselines --races 200 --circuit bahrain
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import ray

from src.rl.build_env_config import build_env_config
from src.rl.eval.model_loader import load_main_module
from src.rl.eval.controllers import ScriptedController, RLController
from src.rl.eval.plans import anchor_plan, load_mc_plan
from src.rl.eval.race_runner import run_race
from src.rl.eval.metrics import win_rate, mean_finish, bootstrap_ci
from src.simulation.regulation_profiles import get_profile

BASELINES = ["anchor_onestop", "anchor_twostop", "mc"]


def _neutral_drivers(drivers):
    """All cars share a representative (median) driver so finish reflects strategy."""
    mid = drivers[len(drivers) // 2]
    return [deepcopy(mid) for _ in drivers]


def main():
    ap = argparse.ArgumentParser(description="RL-3a beat-baselines eval")
    ap.add_argument("--races", type=int, default=200)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--checkpoint", type=str, default="models/rl_league/league")
    ap.add_argument("--module-id", type=str, default="main_1")
    ap.add_argument("--out", type=str, default="results/rl_eval/beat_baselines.json")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    cfg = build_env_config(args.season, args.circuit)
    circuit, drivers = cfg["circuit"], _neutral_drivers(cfg["drivers"])
    profile = get_profile(args.season)
    n = len(drivers)

    module, algo = load_main_module(args.checkpoint, args.module_id)
    mc_start, mc_plan = load_mc_plan(
        f"results/scenarios_{args.circuit}_{args.season}.json", circuit.total_laps)

    def make_controller(kind):
        if kind == "rl":
            return RLController(module)
        if kind == "mc":
            return ScriptedController(mc_start, mc_plan)
        start, plan = anchor_plan(kind.replace("anchor_", ""))
        return ScriptedController(start, plan)

    # finishes[label] = list of finishing positions across races for that controller type
    finishes = {k: [] for k in ["rl", *BASELINES]}
    # paired[b] = (rl_finishes, b_finishes) sampled head-to-head within each race
    paired = {b: ([], []) for b in BASELINES}

    for s in range(args.races):
        rl_slot = s % n
        kinds = [None] * n
        kinds[rl_slot] = "rl"
        others = [i for i in range(n) if i != rl_slot]
        for j, i in enumerate(others):                 # split remaining slots across baselines
            kinds[i] = BASELINES[j % len(BASELINES)]
        controllers = [make_controller(k) for k in kinds]
        pos = run_race(circuit, drivers, controllers, profile, seed=s)
        for i, k in enumerate(kinds):
            finishes[k].append(pos[i])
        rl_pos = pos[rl_slot]
        for b in BASELINES:                            # nearest baseline car of each type
            b_slots = [i for i, k in enumerate(kinds) if k == b]
            if b_slots:
                paired[b][0].append(rl_pos)
                paired[b][1].append(pos[b_slots[0]])

    report = {"circuit": args.circuit, "season": args.season, "races": args.races,
              "module_id": args.module_id,
              "mean_finish": {k: mean_finish(v) for k, v in finishes.items()},
              "mean_finish_ci95": {k: bootstrap_ci(v) for k, v in finishes.items()},
              "rl_winrate_vs": {b: win_rate(paired[b][0], paired[b][1]) for b in BASELINES}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"✓ wrote {out}", flush=True)
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check (laptop)** — `python -m py_compile src/rl/eval/run_beat_baselines.py`.

- [ ] **Step 3: Write sbatch** `scripts/hpc/eval_rl.sbatch`:

```bash
#!/usr/bin/env bash
# RL-3a beat-baselines eval. Submit: sbatch scripts/hpc/eval_rl.sbatch
#SBATCH --job-name=f1-rl3a
#SBATCH --output=logs/rl3a_%j.out
#SBATCH --error=logs/rl3a_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
# #SBATCH --partition=<your_cpu_partition>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/rl_eval
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy
export CUDA_VISIBLE_DEVICES=""
RACES="${RACES:-200}"; CIRCUIT="${CIRCUIT:-bahrain}"
echo "▸ RL-3a eval: races=$RACES circuit=$CIRCUIT"
python -m src.rl.eval.run_beat_baselines --races "$RACES" --circuit "$CIRCUIT"
echo "✓ done"
```

- [ ] **Step 4: HPC smoke** — `python -m src.rl.eval.run_beat_baselines --races 8 --circuit bahrain`.
  Expect a printed JSON with `mean_finish`/`rl_winrate_vs` and `✓ wrote results/rl_eval/beat_baselines.json`,
  no errors. Likely fix spots: `model_loader.get_module` accessor, `RLController` forward key,
  `MultiCarRaceSim(...)` positional args (mirror the env's constructor call exactly).

- [ ] **Step 5: Run the real eval + check the RL-3a gate** — `sbatch scripts/hpc/eval_rl.sbatch`
  (or `RACES=300 sbatch ...`). Inspect `results/rl_eval/beat_baselines.json`:
  - `rl_winrate_vs.anchor_onestop` and `.anchor_twostop` clearly > 0.5 (RL beats the anchors).
  - `mean_finish.rl` lower (better) than the baselines, with **non-overlapping 95% CIs** vs at
    least the anchors.
  - `rl_winrate_vs.mc` reported honestly (this is the hard baseline — whatever it shows is the result).
- [ ] **Step 6: Commit** (skipped — user commits; gitignore large checkpoints, keep the small JSON)

---

## Self-review

**Spec coverage:** controllers (RL/scripted/MC) → Tasks 3,6 + plans Task 2; race over validated sim → Task 7; shared obs (training-identical) → Task 4; model load → Task 5; bootstrap-CI metrics → Task 1; orchestrator + JSON + gate → Task 8; data contract JSON at `results/rl_eval/beat_baselines.json` → Task 8. ✓ (Figures: the JSON is the contract for RL-3d; a static figure is a thin add — folded into Task 8 output as optional, RL-3d renders the real charts.)

**Placeholder scan:** none — full code in every task. The two "API-drift note"s (Tasks 5,6) are explicit expected-adjustment points, consistent with how RL-2 went.

**Type consistency:** `ScriptedController(start_compound, plan)` / `RLController(module)` both expose `.start_compound` + `.decide(obs)->int`, used by `run_race` (Task 7) and `run_beat_baselines` (Task 8). `parse_mc_plan`/`anchor_plan`/`load_mc_plan` (Task 2) return `(start_compound, plan)` consumed in Task 8. `car_obs(sim, circuit, profile, drivers, i)` (Task 4) used by both the env and `run_race`. action ints 0–3 consistent with `ma_obs.action_to_compound` / `legal_action_mask`. `win_rate`/`mean_finish`/`bootstrap_ci` (Task 1) used in Task 8.

**Note (honest):** `run_beat_baselines` figure generation is deferred — Task 8 emits the JSON (the data contract RL-3d needs); publication figures live with the website work in RL-3d, avoiding duplicate plotting. The gate is read from the JSON.
