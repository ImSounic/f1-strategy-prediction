# 🏎️ F1 Race Strategy Optimizer

A data-driven system for optimizing Formula 1 pit stop strategies using machine learning, Monte Carlo simulation, and Bayesian inference.

**University of Twente — Data Science Module**
**Topics: Data Mining · Feature Extraction from Time Series**

---

## Development status

Active "Step 2" effort: position modeling, 2026-regulation readiness, and a
reinforcement-learning workstream.

- **Done & validated:** Phases 0–3.6 (era-profile abstraction, reproducible driver
  configs, finishing-order validation @ Spearman ~0.70, compound differentiation +
  temporal prior) and **Phase 4** (2026 ingested & era-aware, position Spearman 0.71).
- **RL-1 done:** `multi_car_sim` made step-able + RLlib multi-agent env on the
  *validated* physics (`src/rl/`).
- **RL-2 in progress:** AlphaStar-style **league self-play** training, staged
  RL-2a → RL-2b → RL-2c.

**Resume / handoff doc:** [`docs/superpowers/RL2-HANDOFF.md`](docs/superpowers/RL2-HANDOFF.md)
(consolidated state, decisions, next action). Specs & plans live in
`docs/superpowers/specs/` and `docs/superpowers/plans/`.

---

## Key Results

| Metric | Value |
|--------|-------|
| Stop-count accuracy — right number of stops (dry, 2025) | **71% exact** |
| Full-strategy top-5 — sequence in top-5 candidates (dry, 2025) | **52%** |
| Tyre degradation model MAE (2025 fold CV) | **0.082 s/lap** |
| Monte Carlo simulation speed | **~9,000 sims/sec** |

> *Two notions of accuracy:* **stop-count** (did we pick the right number of pit
> stops?) and **full strategy** (did we pick the right ordered compound
> sequence, e.g. MEDIUM → HARD?). Stop count is the easier target, so it scores
> higher. See [`src/analysis/strategy_match.py`](src/analysis/strategy_match.py).

## System Architecture

```
Data Ingestion → Feature Engineering → Model Training → Monte Carlo Simulator → Strategy Recommendation
(4 APIs)         (Savitzky-Golay)      (XGBoost)        (1000 sims/strategy)   (71% accuracy)
```

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| FastF1 | Lap times, weather, track status | 2022–2025 (92 races) |
| OpenF1 | Stint data, pit stops | 2023–2025 |
| Jolpica | Race results, qualifying, standings | 2022–2025 |
| Pirelli | Circuit characteristics (manual collection) | 2022–2025 |

## Models

### 1. Tyre Degradation (XGBoost)
- Predicts degradation rate (s/lap) per stint
- Features: circuit characteristics, weather, compound, stint length
- Compared against Ridge (baseline) and MLP (neural network)
- SHAP analysis for interpretability

### 2. Safety Car Probability (Bayesian Beta-Binomial)
- Per-circuit SC/VSC probabilities with shrinkage to global mean
- Honest approach: RF classifier confirms SC events are fundamentally stochastic (AUC ≈ 0.5)

### 3. Circuit Similarity (Hierarchical Clustering)
- Groups circuits by degradation characteristics
- 2 clusters: High-deg (abrasive, high-speed) vs Low-deg (street, smooth)

### 4. Stint Similarity (Dynamic Time Warping)
- Finds historically similar stints for mid-race strategy adaptation
- Silhouette score: 0.94

## Validation

Rolling temporal validation with **zero data leakage**:

| Fold | Training | CV MAE | Stops Correct | Strategy Top-5 |
|------|----------|--------|---------------|----------------|
| 2022 → 2023 | 967 stints | 0.109s | 47% | 37% |
| 2022-23 → 2024 | 1,982 stints | 0.084s | 59% | 50% |
| 2022-24 → 2025 | 3,006 stints | 0.082s | **71%** | **52%** |

- **Stops Correct** = the top-ranked strategy has the right *number* of pit stops.
- **Strategy Top-5** = the actual *compound sequence* appears in the top-5 distinct
  candidates. These are different metrics, so Stops Correct can exceed Top-5.

> An earlier version of this table reported a "Top-5" of 86%, which was an
> artifact of a metric that compared only stop counts (top-k collapsed onto
> exact match). The numbers above use corrected full-strategy matching
> ([`strategy_match.py`](src/analysis/strategy_match.py), unit-tested). Full-strategy
> accuracy is partly capped by candidate coverage — the generator does not yet
> emit some real sequences (e.g. MEDIUM → HARD → HARD), a known improvement target.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
make all

# Or run individual phases
make ingest      # Phase 1: Data collection
make prepare     # Phase 2: Feature engineering
make model       # Phase 3: Model training
make simulate    # Phase 4: Monte Carlo simulation
make analyze     # Phase 5a: SHAP, DTW, validation
make visualize   # Phase 5b: Report figures
```

## Project Structure

```
f1-strategy-optimizer/
├── configs/
│   └── config.yaml              # Central configuration
├── data/
│   ├── raw/                     # Raw API data
│   │   ├── fastf1/             # Laps, weather, track status
│   │   ├── jolpica/            # Results, pitstops, standings
│   │   ├── openf1/             # Stints, pitstops
│   │   └── supplementary/      # Pirelli circuit characteristics
│   └── features/               # Engineered features
├── models/                      # Trained models & evaluation
├── results/
│   ├── figures/                # Report figures
│   └── *.json                  # Strategy & validation results
├── src/
│   ├── ingestion/              # Data extraction scripts
│   ├── preparation/            # Cleaning & feature engineering
│   ├── modeling/               # Model training & comparison
│   ├── simulation/             # Monte Carlo strategy simulator
│   ├── analysis/               # SHAP, DTW, validation
│   └── visualization/          # Report figure generation
├── Makefile                    # One-command pipeline
├── requirements.txt            # Dependencies
└── README.md
```

## Deployment

| Component | Platform | Tier |
|-----------|----------|------|
| Frontend | Vercel | Free |

## Limitations & Future Work

- **Dry conditions only**: No INTERMEDIATE/WET tyre modeling
- **Max 2-stop strategies**: Some races require 3+ stops
- **No position modeling**: Optimizes total time, not accounting for track position / undercut
- **Static weather**: Doesn't model mid-race weather changes
- **Future**: Live telemetry integration, reinforcement learning for dynamic re-planning, Next.js frontend

## Technology Stack

Python 3.10+ · XGBoost · scikit-learn · SHAP · SciPy · FastF1 · Matplotlib

## References

- Grinsztajn et al. (2022) — "Why do tree-based models still outperform deep learning on tabular data?"
- Bergstra & Bengio (2012) — Random search for hyper-parameter optimization
- Lundberg & Lee (2017) — SHAP: A unified approach to interpreting model predictions
