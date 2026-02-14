# F1 Race Strategy Optimizer

## Quick Orientation

This is a **production-grade F1 race strategy optimization system** built for a University of Twente Data Science module. It combines tyre degradation modeling (XGBoost), Monte Carlo strategy simulation, reinforcement learning (PPO), and a multi-car race simulator with a Next.js frontend deployed on Vercel.

**Start by reading the codebase structure below, then explore the key files to understand the architecture.**

---

## Project Structure

```
f1-strategy-optimizer/
├── src/
│   ├── data/                    # Data ingestion pipeline
│   │   ├── fastf1_client.py     # FastF1 API: lap times, telemetry, tyre stints
│   │   ├── jolpica_client.py    # Jolpica/Ergast API: race results, standings
│   │   ├── openf1_client.py     # OpenF1 API: real-time race data
│   │   └── pirelli_scraper.py   # Circuit tyre compound allocations
│   │
│   ├── features/                # Feature engineering
│   │   └── build_features.py    # Stint-level features for degradation model
│   │
│   ├── models/                  # ML modeling
│   │   ├── train.py             # Ridge/XGBoost/MLP comparison + tuning
│   │   └── evaluate.py          # Temporal validation, SHAP, DTW analysis
│   │
│   ├── simulation/              # Strategy simulation engines
│   │   ├── strategy_simulator.py   # Single-car Monte Carlo simulator (9K sims/sec)
│   │   ├── multi_car_sim.py        # 20-car race engine (overtaking, DRS, SC, team orders)
│   │   └── precompute_scenarios.py # Multi-car precomputation pipeline → frontend data
│   │
│   ├── scripts/                 # Standalone scripts
│   │   ├── precompute_scenarios.py  # Single-car MC for all circuits (→ results/*.json)
│   │   ├── run_pipeline.py          # End-to-end pipeline runner
│   │   └── generate_figures.py      # Report figures
│   │
│   ├── rl/                      # Reinforcement Learning
│   │   ├── f1_env.py            # Gymnasium environment wrapping simulator
│   │   ├── train_agent.py       # PPO training with Stable-Baselines3
│   │   └── evaluate_agent.py    # RL vs MC head-to-head evaluation
│   │
│   └── visualization/           # Plotting utilities
│       └── report_figures.py    # 8 publication-quality figures
│
├── frontend/                    # Next.js 14 frontend (Vercel)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx              # Main analysis page
│   │   │   ├── simulate/page.tsx     # Race simulator page
│   │   │   └── layout.tsx            # Root layout
│   │   ├── components/
│   │   │   ├── Header.tsx            # Navigation (Analysis | Race Simulator tabs)
│   │   │   ├── StrategyDashboard.tsx # MC strategy analysis dashboard
│   │   │   ├── SimulatorView.tsx     # Multi-car simulator UI (driver cards, charts)
│   │   │   ├── ScenarioPlanner.tsx   # Contingency scenario planner
│   │   │   └── RLDashboard.tsx       # RL vs MC comparison dashboard
│   │   └── data/
│   │       ├── strategies.ts         # Pre-computed single-car MC strategies
│   │       ├── scenarios.ts          # Pre-computed multi-car simulation stats
│   │       ├── scenarioResults.ts    # Contingency scenario results
│   │       └── rlResults.ts          # RL evaluation results
│   └── public/
│       └── scenarios/                # Per-circuit detail JSON (loaded on demand)
│           ├── bahrain.json
│           ├── monaco.json
│           └── ... (24 circuits)
│
├── configs/
│   ├── config.yaml              # Master config (seasons, circuits, paths)
│   ├── drivers_2024.json        # 20 drivers: pace deltas, overtaking, tyre management
│   └── circuits/                # Per-circuit config overrides
│
├── models/                      # Trained models
│   ├── tyre_deg_production.json # XGBoost degradation model
│   ├── comparison_results.json  # Model comparison metrics
│   └── rl_agent/                # PPO checkpoints
│
├── results/                     # Output data
│   ├── strategy_*_2025.json     # Per-circuit MC strategy results
│   ├── scenarios/               # Multi-car scenario JSONs
│   ├── rl_evaluation/           # RL vs MC comparison data
│   └── figures/                 # Generated report figures
│
├── data/                        # Raw + processed data
│   ├── raw/                     # FastF1 cache, API responses
│   └── processed/               # Feature matrices, stint data
│
└── CLAUDE.md                    # This file
```

---

## Development History (Chronological)

### Phase 1: Data Ingestion (Feb 9)
- Built FastF1, Jolpica, OpenF1 API clients with rate limiting
- Extracted lap times, tyre stints, pit stops for 2022-2025 seasons
- Generated Pirelli compound allocation data per circuit

### Phase 2: Feature Engineering (Feb 9)
- Stint-level features: fuel load, tyre age, compound hardness, track temperature
- Interaction features: compound×circuit, stint_number×fuel_corrected_delta
- ~15K stint samples across 4 seasons

### Phase 3: Degradation Modeling (Feb 9-10)
- Compared Ridge, XGBoost, MLP → XGBoost selected (best accuracy, interpretable)
- SHAP analysis for feature importance
- DTW stint similarity clustering
- Rolling temporal validation (train on past, predict future)
- **Result**: 71% exact strategy match on 2025 unseen data

### Phase 4: Monte Carlo Strategy Simulator (Feb 10)
- Single-car simulator: tests all viable strategies × 10K simulations
- Accounts for: tyre degradation, pit loss, safety car probability, fuel effect
- Performance: ~9,000 simulations/second
- Outputs ranked strategies with confidence intervals

### Phase 5: Visualization & Deployment (Feb 10)
- 8 publication-quality figures (SHAP, DTW, strategy comparison, etc.)
- FastAPI backend (initially on Render, later removed)
- Next.js frontend on Vercel with static pre-computed data
- Pre-computation pipeline: single-car MC for all 24 circuits

### Phase 6: Frontend Enhancements (Feb 10)
- Methodology & Limitations sections
- Compound color badges (red SOFT, yellow MEDIUM, white HARD)
- Animated hero, racing line SVG
- Strategy table with sortable columns

### Phase 7: Contingency Scenario Planner (Feb 10-11)
- Conditional MC: forces specific conditions (early SC, late SC, high deg, etc.)
- Decision triggers: "If SC before lap X, switch to strategy Y"
- Interactive frontend dashboard

### Phase 8: Reinforcement Learning Agent (Feb 11)
- Gymnasium environment wrapping MC simulator
- PPO training with Stable-Baselines3 (500K timesteps)
- Head-to-head: RL vs MC over 1000 races
- Fixed reward bug causing over-pitting (2.67 stops → 1.2 stops)
- Frontend: win rates, SC breakdown, time distributions, lap-by-lap traces
- Key insight: MC more consistent (faster median), RL adapts better under SC

### Phase 9: Multi-Car Race Simulator (Feb 12)
- Full 20-car field simulation with position-aware optimization
- Overtaking model: DRS (-0.3s within 1s), dirty air (+0.15s within 1.5s)
- Blue flags, team orders, SC field compression
- Greedy SC reactor: pits under SC only if tyres >40% worn
- Pre-computation: 24 circuits × 20 drivers × 20 positions × 50 sims
- Frontend: driver selection cards, position slider, compound-colored charts
- **Latest**: Tabbed race scenarios (Best/Worst/Median/Early SC/Late SC)
  with auto-generated narratives and on-demand detail JSON loading

---

## Key Commands

### Single-Car Monte Carlo (per circuit)
```bash
# One circuit (note: --circuit singular, not --circuits)
python -m src.scripts.precompute_scenarios --circuit bahrain --n-sims 50

# All circuits (loop — this script only accepts one circuit at a time)
for circuit in bahrain jeddah albert_park suzuka shanghai miami imola monaco montreal barcelona spielberg silverstone hungaroring spa zandvoort monza baku singapore cota mexico interlagos las_vegas lusail yas_marina; do
  python -m src.scripts.precompute_scenarios --circuit "$circuit" --n-sims 50
done
```
**Output**: `results/strategy_{circuit}_2025.json` — consumed by multi-car simulator and frontend

### Multi-Car Simulator Precomputation
```bash
# All circuits (takes ~10 hours with 50 sims)
python -m src.simulation.precompute_scenarios --circuits all --n-sims 50

# Single circuit
python -m src.simulation.precompute_scenarios --circuits bahrain --n-sims 50

# Background run
nohup python -m src.simulation.precompute_scenarios --circuits all --n-sims 50 > precompute.log 2>&1 &
```
**Output**: `frontend/src/data/scenarios.ts` (compact stats) + `frontend/public/scenarios/{circuit}.json` (full race detail)

### RL Training & Evaluation
```bash
# Train PPO agent
python -m src.rl.train_agent --circuit bahrain --timesteps 500000

# Evaluate RL vs MC
python -m src.rl.evaluate_agent --circuit bahrain --n-races 1000
```

### Frontend
```bash
cd frontend
npm install
npm run dev          # localhost:3000
npm run build        # Production build (static export for Vercel)
```

### Full Pipeline
```bash
python -m src.scripts.run_pipeline      # End-to-end: data → features → model → simulation
python -m src.scripts.generate_figures   # Report figures → results/figures/
```

### Deployment
```bash
git add -A && git commit -m "feat: description" && git push
# Vercel auto-deploys from main branch
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | FastF1, OpenF1 API, Jolpica/Ergast API |
| ML | XGBoost (degradation), Stable-Baselines3 (PPO) |
| Simulation | NumPy-based MC engine, custom multi-car simulator |
| Backend | Python 3.11+, FastAPI (optional) |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Deployment | Vercel (frontend), GitHub |
| Analysis | SHAP, DTW (tslearn), Matplotlib, Seaborn |

---

## Architecture Decisions

### Why XGBoost over Neural Networks?
- Tabular data with ~15K samples → tree-based models dominate
- SHAP interpretability required for academic submission
- XGBoost matched MLP accuracy with 10x faster training

### Why MC over RL for Strategy Selection?
- MC is more statistically robust: tests ALL strategies exhaustively
- RL adapts better under SC but is less consistent overall
- MC median time ~1-2s faster than RL across 1000 races
- Both approaches kept for academic comparison value

### Why Position-Based Scoring for Multi-Car Sim?
- Single-car sim optimizes total race time (no interaction)
- Multi-car sim must account for overtaking, traffic, DRS effects
- Median finishing position is the correct objective with 20-car interactions
- Strategy that's fastest in isolation may not be best in traffic

### Why Pre-Computation over Live Simulation?
- 9,600 combos × 50 sims × 20 strategies = billions of sim steps
- Pre-compute once (~10 hours), serve instantly
- Frontend fetches per-circuit JSON on demand (lazy loading)
- Compact TS file for stats, detailed JSON for race traces

---

## Data Flow

```
FastF1/OpenF1/Jolpica APIs
    ↓
Raw Data (data/raw/)
    ↓
Feature Engineering (src/features/)
    ↓
XGBoost Degradation Model (models/tyre_deg_production.json)
    ↓
┌──────────────────────────────────────────────────────┐
│              Strategy Simulation Layer                │
│                                                      │
│  Single-Car MC          Multi-Car Sim     RL Agent   │
│  (src/scripts/          (src/simulation/  (src/rl/)  │
│   precompute_scenarios)  precompute_scenarios)       │
│       ↓                      ↓                ↓     │
│  results/*.json    scenarios.ts + .json   rl_eval/   │
└──────────────────────────────────────────────────────┘
    ↓                      ↓                    ↓
┌──────────────────────────────────────────────────────┐
│               Next.js Frontend (Vercel)              │
│                                                      │
│  Strategy Dashboard │ Race Simulator │ RL Dashboard  │
│  Scenario Planner   │ Methodology    │ Limitations   │
└──────────────────────────────────────────────────────┘
```

---

## Current State & Known Issues

### Working
- All 24 circuits pre-computed for single-car MC
- Multi-car simulator precomputed (24 circuits × 20 drivers × 20 positions)
- Frontend deployed on Vercel with all dashboards
- RL agent trained and evaluated for Bahrain
- Tabbed race scenario display (best/worst/median/early SC/late SC)

### Needs Re-run
- Multi-car precompute needs re-run with updated code that captures 5 sample races
  per combo (best/worst/median/early SC/late SC) instead of just 1
- This generates both `frontend/src/data/scenarios.ts` and `frontend/public/scenarios/{circuit}.json`
- Run: `nohup python -m src.simulation.precompute_scenarios --circuits all --n-sims 50 > precompute.log 2>&1 &`

### Known Limitations
- No wet weather modeling (dry conditions only)
- 2-stop strategies max in some scenarios (3+ stop rarely tested)
- SC timing is stochastic, not predictive
- Driver performance assumed constant through race (no fatigue)
- No qualifying simulation (grid position is input, not predicted)
- Monaco 2-stop anomaly: simulator sometimes prefers 2-stop due to position modeling gaps

---

## Important File Details

### `src/simulation/multi_car_sim.py` (~450 lines)
Core 20-car race engine. Key classes:
- `DriverConfig`: pace_delta, overtaking skill, tyre_management
- `CircuitParams`: laps, pit_loss, SC probability, overtaking difficulty
- `Strategy`: compound sequence with stint lengths
- `MultiCarRaceSim`: lap-by-lap simulation with overtaking, DRS, SC
- `generate_common_strategies()`: creates ~20 viable strategy candidates
- `build_grid()` / `find_target_in_grid()`: positions target driver in 20-car field

### `src/simulation/precompute_scenarios.py` (~650 lines)
Pipeline that runs multi-car sim for all combos. Key functions:
- `evaluate_strategy()`: runs N sims, picks 5 representative races (best/worst/median/early SC/late SC)
- `optimize_scenario()`: tests top 20 strategies, narrows to 8, full-evaluates best
- `write_typescript()`: compact stats → `scenarios.ts`
- `write_detail_json()`: full race traces → `public/scenarios/{circuit}.json`
- `_generate_narrative()`: auto-writes race description text
- `load_circuit_as_params()`: converts existing CircuitConfig → CircuitParams with XGBoost deg rates
- `get_available_circuits()`: scans `results/strategy_*_2025.json` for available circuits

### `src/scripts/precompute_scenarios.py`
Single-car MC optimizer. CLI: `--circuit <name> --n-sims N --season YYYY --config path`
Note: this is DIFFERENT from `src/simulation/precompute_scenarios.py` (multi-car).

### `frontend/src/components/SimulatorView.tsx` (~600 lines)
Race simulator UI. Features:
- 20 driver cards with team colors and numbers
- 24 circuit dropdown with flag emoji
- P1-P20 grid position slider
- Tabbed scenarios: Best Case / Median / Worst Case / Early SC / Late SC
- Position chart with compound-colored stint backgrounds, pit markers, SC lines
- Tyre age mini chart
- Auto-generated narrative per scenario
- On-demand JSON loading per circuit (lazy, not bundled)

### `configs/drivers_2024.json`
20 F1 drivers with:
- `pace_delta`: seconds vs reference lap (VER fastest at -0.5, SAR slowest at +1.2)
- `overtaking`: 0-1 skill (VER 0.95, SAR 0.30)
- `tyre_management`: 0-1 skill (HAM 0.95, MAG 0.55)
- `team`: for team orders logic
- `teammate_code`: for team orders pairing

---

## Tailwind Custom Theme (frontend)

The frontend uses a dark F1-inspired theme with custom color tokens:
- `f1-dark`: #0a0a1a (page background)
- `f1-card`: #111125 (card background)
- `f1-darker`: #0d0d20 (input backgrounds)
- `f1-border`: #1e1e3a (borders)
- `f1-muted`: #8888aa (secondary text)
- `f1-red`: #e10600 (F1 red accent)

Custom fonts: `font-display` (bold headings), `font-mono` (data/stats), `font-body` (paragraphs)

---

## For Claude Code: First Steps

1. **Read this file** to understand the full architecture
2. **Explore the codebase**: start with `src/simulation/multi_car_sim.py` and `frontend/src/components/SimulatorView.tsx`
3. **Check current data state**:
   - `ls results/strategy_*_2025.json | wc -l` (should be ~24 single-car MC results)
   - `ls frontend/src/data/scenarios.ts` (multi-car stats)
   - `ls frontend/public/scenarios/` (multi-car detail JSONs)
4. **Test locally**: `cd frontend && npm run dev` → localhost:3000
5. **Test simulation**: `python -c "from src.simulation.multi_car_sim import *; print('OK')"`

### When making changes:
- Backend simulation: test with `--circuits bahrain --n-sims 10` first (fast iteration)
- Frontend: `npm run dev` and verify in browser before committing
- Always commit and push after verified changes
- The Vercel deployment auto-triggers on push to mainrred