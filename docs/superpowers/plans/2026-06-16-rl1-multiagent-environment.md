# RL-1 — Multi-Agent Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the validated `multi_car_sim` step-able and wrap it in an RLlib `MultiAgentEnv` (richer pit+compound actions, driver-conditioned observations, finishing-position reward) — without changing the validated physics.

**Architecture:** Extract the per-lap loop of `MultiCarRaceSim.run()` into `reset()` + `step(pit_override)`; `run()` becomes a thin loop calling them, so behaviour is preserved *by construction* and proven by a golden test + a Phase 2/3 re-run. Pure obs/reward/action helpers live in a dependency-free module (laptop-testable). The RLlib env (needs Ray) is the only HPC-only piece.

**Tech Stack:** Python 3.11, numpy (sim), Ray RLlib (env conformance only in RL-1). Pure helpers + golden test run on the laptop; RLlib env smoke-test on HPC.

---

## Scope
RL-1 is **only the environment** (step-able sim + pure helpers + RLlib env + tests). League self-play (RL-2), training, and evaluation (RL-3) are separate. The RL-1 acceptance gate is: **Phase 2/3 validation Spearman unchanged (~0.70)** after the refactor.

## File structure
```
src/rl/ma_obs.py                       # NEW — pure helpers (action->compound, masking, reward, obs)
src/simulation/multi_car_sim.py        # MODIFY — extract reset()/step(); run() loops over them
src/rl/multiagent_env.py               # NEW — RLlib MultiAgentEnv on the step-able sim
tests/test_ma_obs.py                   # NEW — pure unit tests (laptop)
tests/test_multi_car_sim_step.py       # NEW — golden/regression for the step refactor (laptop, numpy)
```

---

## Task 1: Pure env helpers (`ma_obs.py`)

**Files:** Create `src/rl/ma_obs.py`; Test `tests/test_ma_obs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ma_obs.py`:

```python
"""Pure tests for multi-agent RL helpers. Run: python tests/test_ma_obs.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.ma_obs import action_to_compound, legal_action_mask, reward_from_positions, build_obs, OBS_DIM


def test_action_to_compound():
    assert action_to_compound(0) is None          # stay
    assert action_to_compound(1) == "SOFT"
    assert action_to_compound(2) == "MEDIUM"
    assert action_to_compound(3) == "HARD"


def test_legal_mask_stay_always_legal():
    m = legal_action_mask(stops_done=3, tyre_age=1, current_lap=1, total_laps=50)
    assert m[0] is True and m[1:] == [False, False, False]   # max stops + lap1 + young tyre


def test_legal_mask_allows_pit_midrace():
    m = legal_action_mask(stops_done=0, tyre_age=10, current_lap=20, total_laps=50)
    assert m == [True, True, True, True]


def test_legal_mask_blocks_late_and_young():
    assert legal_action_mask(0, 10, 49, 50)[1:] == [False, False, False]  # too late
    assert legal_action_mask(0, 2, 20, 50)[1:] == [False, False, False]   # tyre too young


def test_reward_positions_gained():
    assert reward_from_positions(grid=10, finish=4) == 6.0   # gained 6
    assert reward_from_positions(grid=1, finish=3) == -2.0   # lost 2


def test_build_obs_shape_and_bounds():
    state = dict(lap=10, total_laps=50, tyre_age=8, compound_idx=2, cumulative_deg=1.2,
                 position=5, n_cars=20, gap_ahead=1.3, gap_behind=0.8, fuel_frac=0.6,
                 sc_active=0, stops_done=1, max_stops=3, compounds_used=2, laps_since_sc=12,
                 driver_pace=0.3, driver_overtaking=0.6, driver_tyre=0.7,
                 pit_loss=23.0, sc_prob=0.55, overtaking_difficulty=0.5)
    obs = build_obs(state)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype.name == "float32"
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.5


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

Run: `python tests/test_ma_obs.py`
Expected: `ModuleNotFoundError: No module named 'src.rl.ma_obs'`

- [ ] **Step 3: Write the module**

Create `src/rl/ma_obs.py`:

```python
"""
Pure helpers for the multi-agent F1 RL environment
==================================================
No Ray / no gym / no sim imports — unit-testable on any machine. The RLlib env
(multiagent_env.py) composes these to build observations, mask illegal actions,
translate actions to pit decisions, and compute rewards.
"""
from __future__ import annotations

import numpy as np

# Discrete(4) action -> compound to switch to (None = stay out)
_ACTION_COMPOUND = {0: None, 1: "SOFT", 2: "MEDIUM", 3: "HARD"}

MIN_STINT_LAPS = 3   # no re-pit before the tyre has this many laps
OBS_DIM = 18


def action_to_compound(action: int):
    """Map a Discrete(4) action to a compound name, or None for 'stay out'."""
    return _ACTION_COMPOUND[int(action)]


def legal_action_mask(stops_done: int, tyre_age: int, current_lap: int,
                      total_laps: int, max_stops: int = 3,
                      min_stint: int = MIN_STINT_LAPS) -> list:
    """Boolean mask over [stay, pit-S, pit-M, pit-H]. Stay is always legal.

    Pitting is illegal if max stops reached, the tyre is too young, or it's the
    last lap or two.
    """
    can_pit = (stops_done < max_stops
               and tyre_age >= min_stint
               and current_lap < total_laps - 1)
    return [True, can_pit, can_pit, can_pit]


def reward_from_positions(grid: int, finish: int) -> float:
    """Terminal reward = positions gained (grid - finish). Higher is better."""
    return float(grid - finish)


def build_obs(state: dict) -> np.ndarray:
    """Build the normalised, driver-conditioned observation vector (OBS_DIM,)."""
    obs = np.array([
        state["lap"] / max(state["total_laps"], 1),
        min(state["tyre_age"], 50) / 50.0,
        (state["compound_idx"] + 1) / 3.0,
        min(state["cumulative_deg"], 10.0) / 10.0,
        state["position"] / max(state["n_cars"], 1),
        min(state["gap_ahead"], 5.0) / 5.0,
        min(state["gap_behind"], 5.0) / 5.0,
        state["fuel_frac"],
        float(state["sc_active"]),
        state["stops_done"] / max(state["max_stops"], 1),
        state["compounds_used"] / 3.0,
        min(state["laps_since_sc"], 20) / 20.0,
        state["driver_pace"],
        state["driver_overtaking"],
        state["driver_tyre"],
        state["pit_loss"] / 30.0,
        state["sc_prob"],
        state["overtaking_difficulty"],
    ], dtype=np.float32)
    return np.clip(obs, 0.0, 1.5)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python tests/test_ma_obs.py`
Expected: `6/6 passed`

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Task 2: Make `multi_car_sim` step-able (behaviour-preserving)

**Files:** Modify `src/simulation/multi_car_sim.py`; Test `tests/test_multi_car_sim_step.py`

The refactor extracts the body of `run()`'s `for lap` loop into `step(pit_override=None)`,
with race state held on `self`. `run()` becomes `reset()` + a loop of `step()`. A golden
test captures the *current* `run()` output first and asserts the refactor reproduces it
exactly; Phase 2/3 re-run is the deeper gate.

- [ ] **Step 1: Write the golden characterization test (captures CURRENT behaviour)**

Create `tests/test_multi_car_sim_step.py`:

```python
"""Golden regression for the step-able refactor of multi_car_sim.
Needs numpy only. Run: python tests/test_multi_car_sim_step.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.multi_car_sim import (
    MultiCarRaceSim, DriverConfig, CircuitParams, Strategy,
)


def _scenario():
    drivers = [
        DriverConfig(f"D{i}", f"D{i}", "t" + str(i // 2), 0.05 * i, 0.5, 0.7, "")
        for i in range(6)
    ]
    circuit = CircuitParams(
        circuit_key="t", circuit_name="T", total_laps=30,
        pit_loss_seconds=22.0, sc_prob_per_race=0.4, vsc_prob_per_race=0.2,
        overtaking_difficulty=0.5, deg_rates={},
    )
    strat = Strategy(stints=[("MEDIUM", 15), ("HARD", 15)], name="MH")
    return circuit, drivers, [strat] * 6, strat


def test_run_is_deterministic_and_stable():
    """Golden: a fixed scenario+seed must give a stable result. After the
    reset()/step() refactor this MUST remain identical (physics preserved)."""
    circuit, drivers, strategies, strat = _scenario()
    r1 = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False).run(seed=123)
    r2 = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False).run(seed=123)
    assert r1["finishing_positions"] == r2["finishing_positions"]
    assert r1["target_time"] == r2["target_time"]
    # Permutation sanity
    assert sorted(r1["finishing_positions"]) == list(range(1, 7))


def test_reset_step_matches_run():
    """Driving the field through reset()+step() with the same fixed strategies
    must reproduce run() exactly."""
    circuit, drivers, strategies, strat = _scenario()
    ref = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False).run(seed=123)

    sim = MultiCarRaceSim(circuit, drivers, strategies, 0, strat, greedy_sc=False)
    sim.reset(seed=123)
    while not sim.done:
        sim.step()           # no override -> uses each car's fixed strategy
    res = sim.results()
    assert res["finishing_positions"] == ref["finishing_positions"]
    assert res["target_time"] == ref["target_time"]


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

- [ ] **Step 2: Run to confirm `test_run_is_deterministic_and_stable` passes and `test_reset_step_matches_run` fails**

Run: `python tests/test_multi_car_sim_step.py`
Expected: `test_run_is_deterministic_and_stable` PASS (current `run()` is deterministic);
`test_reset_step_matches_run` ERROR (`reset`/`step`/`done`/`results` don't exist yet).

- [ ] **Step 3: Refactor `MultiCarRaceSim` — extract `reset()` / `step()` / `results()`**

In `src/simulation/multi_car_sim.py`, replace the `run(self, seed=42)` method with the
following four methods. The per-lap body is moved verbatim from the old `run()` loop into
`step()`; the only new logic is the `pit_override` branch in the per-car loop and the
`forced_compound` plumbing. `run()` now just drives `step()`.

```python
    def reset(self, seed: int = 42):
        """Initialise a race; ready for step()."""
        self.rng = np.random.default_rng(seed)
        self.n_laps = self.circuit.total_laps
        self.cars = []
        for i, driver in enumerate(self.drivers):
            s = self.strategies[i] if i != self.target_idx else self.target_strategy
            car = CarState(driver_idx=i, tyre_compound=s.stints[0][0], strategy=s)
            car.compounds_used = {s.stints[0][0]}
            car.cumulative_time = i * 0.8
            self.cars.append(car)
        self.positions = self._update_positions(self.cars)
        self.position_history = np.zeros((self.n_laps, self.n_cars), dtype=int)
        self.sc_laps, self.vsc_laps, self.pit_events = [], [], []
        self.target_history = {"lap_times": [], "compounds": [], "tyre_ages": [],
                               "positions": [], "pit_laps": [], "sc_laps": []}
        self.sc_active = self.vsc_active = False
        self.sc_remaining = self.vsc_remaining = 0
        self.lap = 0
        self.done = False
        return self.positions

    def step(self, pit_override: dict = None):
        """Advance one lap. pit_override maps car_idx -> compound name (pit) or
        None (stay). Cars absent from the dict use their own strategy / greedy SC
        logic (this is how run() reproduces the original behaviour exactly)."""
        lap = self.lap + 1
        self.lap = lap

        # ── SC/VSC state machine (identical to the original run() loop top) ──
        sc_just_started = False
        if self.sc_remaining > 0:
            self.sc_remaining -= 1
            self.sc_active = True
            self.sc_laps.append(lap)
        elif self.vsc_remaining > 0:
            self.vsc_remaining -= 1
            self.vsc_active = True
            self.vsc_laps.append(lap)
        else:
            self.sc_active = False
            self.vsc_active = False
            if self.rng.random() < self.sc_prob_per_lap and 1 < lap < self.n_laps - 3:
                self.sc_remaining = int(self.rng.integers(3, 7))
                self.sc_active = True
                sc_just_started = True
                self.sc_laps.append(lap)
            elif self.rng.random() < self.vsc_prob_per_lap and 1 < lap < self.n_laps - 2:
                self.vsc_remaining = int(self.rng.integers(2, 5))
                self.vsc_active = True
                self.vsc_laps.append(lap)

        if sc_just_started:
            self._compress_field_sc(self.cars, self.positions)

        gaps = self._compute_gaps(self.cars, self.positions)

        for i, (car, driver) in enumerate(zip(self.cars, self.drivers)):
            car.tyre_age += 1

            forced_compound = None
            if pit_override is not None and i in pit_override:
                chosen = pit_override[i]
                should_pit = (chosen is not None and car.stops_done < 3
                              and car.tyre_age >= 3 and lap < self.n_laps)
                forced_compound = chosen
            elif i == self.target_idx and self.greedy_sc:
                should_pit = self._should_pit_strategy(car, lap)
                if not should_pit and (self.sc_active or self.vsc_active):
                    should_pit = self._should_pit_greedy_sc(
                        car, lap, self.sc_active or self.vsc_active,
                        self.positions, gaps, self.rng)
            else:
                should_pit = self._should_pit_strategy(car, lap)

            pit_cost = 0.0
            if should_pit and lap < self.n_laps:
                pit_cost = self._process_pit_stop(
                    car, driver, lap, self.sc_active, self.vsc_active, self.rng,
                    forced_compound=forced_compound)
                self.pit_events.append({"lap": lap, "driver_idx": i,
                                        "compound": car.tyre_compound})
                if i == self.target_idx:
                    self.target_history["pit_laps"].append(lap)

            dirty_air = (gaps[i] < self.profile.dirty_air_window
                         and not self.sc_active and not self.vsc_active)
            lap_time = self._compute_lap_time(
                car, driver, lap, self.sc_active, self.vsc_active, self.rng,
                gap_to_ahead=gaps[i], dirty_air=dirty_air)
            lap_time += pit_cost
            car.cumulative_time += lap_time

            if i == self.target_idx:
                self.target_history["lap_times"].append(round(lap_time, 2))
                self.target_history["compounds"].append(car.tyre_compound)
                self.target_history["tyre_ages"].append(car.tyre_age)
                if lap in self.sc_laps:
                    self.target_history["sc_laps"].append(lap)

        self.positions = self._update_positions(self.cars)
        self.position_history[lap - 1] = self.positions
        self._check_lapped(self.cars, self.positions, lap)
        if not self.sc_active and not self.vsc_active:
            gaps = self._compute_gaps(self.cars, self.positions)
            self._process_overtaking(self.cars, self.positions, gaps, lap, self.rng)
            self.positions = self._update_positions(self.cars)
            self.position_history[lap - 1] = self.positions
        self.target_history["positions"].append(int(self.positions[self.target_idx]))

        if lap >= self.n_laps:
            self.done = True
        return self.positions

    def results(self) -> dict:
        """Assemble the result dict (same shape as the old run() return)."""
        final_positions = self.positions
        return {
            "finishing_positions": final_positions.tolist(),
            "position_history": self.position_history.tolist(),
            "target_position": int(final_positions[self.target_idx]),
            "target_time": round(self.cars[self.target_idx].cumulative_time, 1),
            "sc_laps": self.sc_laps,
            "vsc_laps": self.vsc_laps,
            "pit_events": self.pit_events,
            "target_history": self.target_history,
            "n_sc_events": len(set(self.sc_laps)),
        }

    def run(self, seed: int = 42) -> dict:
        """Full race = reset() then step() every lap. Behaviour identical to the
        original loop (same code path)."""
        self.reset(seed)
        while not self.done:
            self.step()
        return self.results()
```

- [ ] **Step 4: Add `forced_compound` to `_process_pit_stop`**

In `_process_pit_stop`, change the signature and the compound-selection block:

```python
    def _process_pit_stop(
        self, car: CarState, driver: DriverConfig,
        current_lap: int, sc_active: bool, vsc_active: bool,
        rng: np.random.Generator, forced_compound: str = None,
    ) -> float:
```
and replace the "Determine next compound" block with:
```python
        # Determine next compound
        if forced_compound is not None:
            next_compound = forced_compound
        elif car.next_pit_idx < len(car.strategy.stints) - 1:
            next_compound = car.strategy.stints[car.next_pit_idx + 1][0]
        else:
            unused = [c for c in COMPOUNDS if c not in car.compounds_used]
            next_compound = unused[-1] if unused else "HARD"
```

(The rest of `_process_pit_stop` — pit cost, state updates, `car.next_pit_idx += 1` — is unchanged.)

- [ ] **Step 5: Run the golden tests**

Run: `python tests/test_multi_car_sim_step.py`
Expected: `2/2 passed` (both determinism and `reset/step` == `run`).

- [ ] **Step 6: Run the existing sim tests (no behaviour drift)**

Run: `python tests/test_multi_car_sim_profile.py`
Expected: all pass (the profile/compound tests still hold).

- [ ] **Step 7 (HPC): the real gate — Phase 2/3 validation unchanged**

```bash
make position-validate && make position-strategy
python -c "import json; print(json.load(open('results/position_validation_report.json'))['overall'])"
```
Expected: overall `mean_spearman` ≈ 0.70 (unchanged from before the refactor). If it moved materially, the refactor changed physics — stop and diff.

- [ ] **Step 8: Commit** (skipped — user commits at end)

---

## Task 3: RLlib `MultiAgentEnv`

**Files:** Create `src/rl/multiagent_env.py` (RLlib needs Ray → smoke-tested on HPC).

- [ ] **Step 1: Write the env**

Create `src/rl/multiagent_env.py`:

```python
"""
Multi-agent F1 strategy environment (RLlib)
===========================================
Every car is an agent sharing one driver-conditioned policy. Each step advances
the validated multi_car_sim by one lap, applying each agent's pit/compound action.
Reward is terminal (positions gained). Built on the step-able MultiCarRaceSim so
the agents train in the SAME physics validated in Phase 2/3.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy
from src.simulation.regulation_profiles import get_profile
from src.rl.ma_obs import (
    action_to_compound, legal_action_mask, reward_from_positions, build_obs, OBS_DIM,
)

_COMPOUND_IDX = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}


class F1MultiAgentEnv(MultiAgentEnv):
    """N cars, shared policy. agent ids are 'car_0' ... 'car_{N-1}'."""

    def __init__(self, config: dict):
        super().__init__()
        self.circuit = config["circuit"]
        self.drivers = config["drivers"]            # list[DriverConfig], grid order
        self.season = config.get("season", 2025)
        self.profile = get_profile(self.season)
        self.n_cars = len(self.drivers)
        self.agents = [f"car_{i}" for i in range(self.n_cars)]
        self.possible_agents = list(self.agents)
        self.grid = {f"car_{i}": i + 1 for i in range(self.n_cars)}  # grid pos (1-indexed)

        self.observation_space = spaces.Box(0.0, 1.5, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._default_strat = Strategy(stints=[("MEDIUM", self.circuit.total_laps)], name="x")

    def _obs_for(self, i: int) -> np.ndarray:
        car = self.sim.cars[i]
        driver = self.drivers[i]
        gaps = self.sim._compute_gaps(self.sim.cars, self.sim.positions)
        # gap behind: find car directly behind in position order
        order = list(np.argsort(self.sim.positions))
        pos_idx = order.index(i)
        gap_behind = gaps[order[pos_idx + 1]] if pos_idx + 1 < self.n_cars else 999.0
        state = dict(
            lap=self.sim.lap, total_laps=self.circuit.total_laps,
            tyre_age=car.tyre_age, compound_idx=_COMPOUND_IDX.get(car.tyre_compound, 1),
            cumulative_deg=0.0, position=int(self.sim.positions[i]), n_cars=self.n_cars,
            gap_ahead=float(gaps[i]), gap_behind=float(gap_behind),
            fuel_frac=max(0.0, 1.0 - self.sim.burn_rate * self.sim.lap / max(self.profile.start_fuel, 1)),
            sc_active=int(self.sim.sc_active or self.sim.vsc_active),
            stops_done=car.stops_done, max_stops=3, compounds_used=len(car.compounds_used),
            laps_since_sc=0,
            driver_pace=min(driver.pace_delta, 2.0) / 2.0,
            driver_overtaking=driver.overtaking, driver_tyre=driver.tyre_management,
            pit_loss=self.circuit.pit_loss_seconds, sc_prob=self.circuit.sc_prob_per_race,
            overtaking_difficulty=self.circuit.overtaking_difficulty,
        )
        return build_obs(state)

    def reset(self, *, seed=None, options=None):
        strategies = [self._default_strat] * self.n_cars
        self.sim = MultiCarRaceSim(self.circuit, self.drivers, strategies, 0,
                                   self._default_strat, greedy_sc=False, profile=self.profile)
        self.sim.reset(seed=seed if seed is not None else 0)
        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        return obs, {a: {} for a in self.agents}

    def step(self, action_dict: dict):
        pit_override = {}
        for i, a in enumerate(self.agents):
            act = int(action_dict.get(a, 0))
            car = self.sim.cars[i]
            mask = legal_action_mask(car.stops_done, car.tyre_age, self.sim.lap + 1,
                                     self.circuit.total_laps)
            comp = action_to_compound(act) if mask[act] else None
            pit_override[i] = comp
        self.sim.step(pit_override)

        done = self.sim.done
        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        if not done:
            rewards = {a: 0.0 for a in self.agents}
        else:
            rewards = {}
            for i, a in enumerate(self.agents):
                finish = int(self.sim.positions[i])
                r = reward_from_positions(self.grid[a], finish)
                # FIA 2-compound rule: penalise illegal single-compound races
                if len(self.sim.cars[i].compounds_used) < 2:
                    r -= 10.0
                rewards[a] = r
        terminateds = {a: done for a in self.agents}
        terminateds["__all__"] = done
        truncateds = {a: False for a in self.agents}
        truncateds["__all__"] = False
        return obs, rewards, terminateds, truncateds, {a: {} for a in self.agents}
```

- [ ] **Step 2: Syntax-check (laptop)**

Run: `python -m py_compile src/rl/multiagent_env.py src/rl/ma_obs.py`
Expected: no output (note: importing the module needs Ray; py_compile only checks syntax).

- [ ] **Step 3 (HPC): Install Ray RLlib if missing**

Run: `python -c "import ray; from ray.rllib.env.multi_agent_env import MultiAgentEnv; print('ray', ray.__version__)"`
If it errors: `pip install "ray[rllib]"` (in the `f1-strategy` env), then re-check.

- [ ] **Step 4 (HPC): Env smoke test — one random episode**

Run:
```bash
python - <<'EOF'
import pandas as pd, yaml
from pathlib import Path
from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params
from src.rl.multiagent_env import F1MultiAgentEnv
import json, xgboost as xgb
cfg=yaml.safe_load(open('configs/config.yaml')); raw=cfg['paths']['raw']
drivers,_,ov = load_drivers('configs/drivers_2025.json')
deg=xgb.XGBRegressor(); deg.load_model('models/tyre_deg_production.json')
fc=json.load(open('models/comparison_results.json'))['experiment']['feature_columns']
circ=load_circuit_as_params('bahrain',2025,cfg,ov,deg,fc)
env=F1MultiAgentEnv({"circuit":circ,"drivers":drivers,"season":2025})
obs,_=env.reset(seed=0)
import numpy as np
done=False; steps=0
while not done:
    acts={a:int(np.random.randint(4)) for a in env.agents}
    obs,rew,term,trunc,_=env.step(acts); done=term["__all__"]; steps+=1
print("episode ok: steps",steps,"sample reward", {k:rew[k] for k in list(rew)[:3]})
print("reward sum", sum(rew.values()), "(should be ~0: positions gained zero-sum)")
EOF
```
Expected: runs ~`total_laps` steps, terminal rewards present, reward sum ≈ 0 (positions gained is zero-sum across the field minus FIA penalties).

- [ ] **Step 5: Commit** (skipped — user commits at end)

---

## Self-review

**Spec coverage (RL-1):**
- Step-able sim sharing validated physics → Task 2 (reset/step/results + run() drives them). ✓
- Regression guard (golden + Phase 2/3 unchanged) → Task 2 Steps 5–7. ✓
- RLlib MultiAgentEnv, driver-conditioned obs, Discrete(4) + masking, terminal position reward, era-aware → Task 3 + Task 1 helpers. ✓
- Pure laptop-testable helpers → Task 1. ✓

**Placeholder scan:** none — full code for helpers, env, and the refactored methods; the refactor reproduces the existing per-lap body with only the `pit_override`/`forced_compound` additions.

**Type consistency:** `OBS_DIM`, `action_to_compound`, `legal_action_mask`, `reward_from_positions`, `build_obs(state)` signatures match across `ma_obs.py`, tests, and `multiagent_env.py`. `MultiCarRaceSim.reset/step/results/done` used consistently in the golden test and the env. `step(pit_override)` dict semantics (idx→compound|None) match `_process_pit_stop(forced_compound=...)`. `profile` attribute already exists on the sim (Phase 0).
