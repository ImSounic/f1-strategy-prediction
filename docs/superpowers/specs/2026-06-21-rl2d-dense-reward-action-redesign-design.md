# Design: RL-2d — Dense Reward + Action Redesign

**Date:** 2026-06-21
**Status:** Approved (design); spec for review
**Parent:** RL-2b league; motivated by the RL-3a finding (agent over-pits, 3 stops, loses to baselines)

---

## 1. Background

RL-3a evaluation (n=200, pace-fair, two fields) showed the trained league agent executes an
aggressive **3-stop** and loses to simple 1-stop/2-stop baselines. Diagnosis confirmed
**structural**, not penalty-size: raising the per-stop penalty PIT_COST 1.0 → 3.0 lowered the
agent's reward (return 0 → −33) but did **not** change the 3-stop behaviour. Root cause: a
**diffuse per-lap pit policy** — greedy → 0 stops, sampled → 3 — because the only learning
signal was a **terminal** reward, giving almost no credit assignment to any individual pit
decision. RL-2d fixes this at the source with a dense reward and a bounded action space.

## 2. Decisions (from brainstorming; user delegated final tuning)

| Decision | Choice |
|---|---|
| Dense reward | **Potential-based shaping**, Φ(s) = −(race position); per-lap reward = Φ(s′)−Φ(s) = `pos_prev − pos_now`. Telescopes to terminal positions-gained (same objective, dense credit). |
| Action space | Keep per-lap `Discrete(4)` (full safety-car reactivity) but tighten the legality mask: **`max_stops = 2`**, **`MIN_STINT_LAPS = 8`** (longer post-pit cooldown). |
| Per-stop cost | **`PIT_COST = 0.5`** (lowered from 3.0): the cap now bounds over-pitting, so the penalty only gently favours 1- over 2-stop; the dense reward decides when a 2nd stop pays. |
| DSQ | **−50** terminal if a dry race used <2 compounds (unchanged; still dominates any legal finish). |
| Reuse | League trainer (RL-2b) and eval harness (RL-3a) unchanged — only the env reward + mask change; retrain + re-eval. |

## 3. Reward specification (the core change)

Per car i, **each lap** (after `sim.step`):
```
r_i = (pos_prev_i − pos_now_i)                      # potential-based shaping (positions gained this lap)
      − PIT_COST  if car i completed a stop beyond its first this lap, else 0
```
At the **terminal** lap, additionally:
```
r_i += −50 (DSQ_PENALTY)  if len(compounds_used_i) < 2
```
Summed over the race this equals `positions-gained − PIT_COST·max(0, n_stops−1) − (50 if DSQ)`
— **identical to the current RL-2 reward, just distributed per-lap.** Same objective; learnable
credit. `pos` is the 1-indexed race position from `sim.positions` (lower = better), so gaining a
place yields a positive lap reward and losing one a negative — immediate signal for good/bad pits.

`pos_prev` is captured before each `sim.step`; `pos_now` after. Extra-stop detection compares
`car.stops_done` before vs after the step (charge when it crosses to ≥2).

## 4. Components / files

- **`src/rl/ma_obs.py`**
  - `legal_action_mask(..., max_stops=2, min_stint=MIN_STINT_LAPS)` — change default `max_stops`
    3 → 2; `MIN_STINT_LAPS` 3 → 8.
  - new pure `shaping_reward(pos_prev, pos_now) -> float` = `pos_prev − pos_now`.
  - keep `terminal_reward`, `DSQ_PENALTY=−50`, `PIT_COST` (set 0.5).
- **`src/rl/multiagent_env.py`**
  - `reset`: store `self._prev_pos = list(self.sim.positions)`.
  - `step`: capture positions before/after; per car reward = `shaping_reward(prev,now)` minus
    `PIT_COST` for an extra stop completed this lap; at `done` add `DSQ_PENALTY` if illegal.
    Update `self._prev_pos`. `build_obs` state uses `max_stops=2`.
- **Tests (pure, laptop):**
  - `test_ma_obs.py`: mask now `max_stops=2` (3 stops → pit illegal); `shaping_reward` sums to
    positions-gained over a position trajectory; `MIN_STINT_LAPS` blocks an early re-pit.

## 5. Data flow
`reset` snapshots positions → each `step`: build per-agent actions → `sim.step(pit_override)` →
read new positions → dense reward per car (shaping − extra-stop cost) → at done add DSQ → update
prev positions. Obs/mask path unchanged otherwise. PPO sees a dense reward stream that still
sums to the same return.

## 6. Testing
- Pure (laptop): mask `max_stops=2` + `min_stint=8` behaviour; `shaping_reward` telescoping
  (Σ over a trajectory grid→finish == grid−finish); DSQ/PIT_COST unchanged tests still pass.
- HPC smoke: `train_league --iters 2 --workers 2` runs with the new env (reward stream non-zero
  per lap; no errors). Then a fresh full retrain.

## 7. Training & gate
Fresh league retrain (~3h, `scripts/hpc/train_league.sbatch`), then re-run RL-3a
(`run_beat_baselines --races 200 --field both`). **Gate:**
- `executed_strategies.rl` shows **1–2 stops** (not 3), and
- **`[real].rl_winrate_vs.* > 0.5`** with `mean_finish.rl` better than the baselines (separated CIs).
If the dense reward + cap still fail to yield a competitive 1–2 stop agent, stop and bank the
honest analysis (RL-3c) rather than iterate further.

## 8. Risks
- **γ residual:** using the pure Φ-difference (γ=1 telescoping) for the dense term while PPO
  discounts at 0.99 is intentional and benign — the dense term is exactly "positions gained per
  lap" and sums to positions-gained; PPO's own discounting handles temporal credit.
- **Cap too tight:** if 2 stops is genuinely optimal somewhere and the agent wants 3, max_stops=2
  forbids it — acceptable (real dry races are 1–2 stops; this is a sound domain constraint).
- **Still diffuse:** dense reward is the principled fix, but if the policy stays diffuse, that
  bounds how far reward shaping alone can go → analysis path.

## 9. Reuse note
No changes to `league.py`, `train_league.py`, `scripted_anchors.py`, `league_callbacks.py`, or
the `src/rl/eval/` harness. RL-2d is purely an env reward/mask revision + retrain + re-eval.
