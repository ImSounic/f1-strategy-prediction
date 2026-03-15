# 🏎️ F1 Race Strategy Optimizer

A data-driven system for optimizing Formula 1 pit stop strategies using machine learning, Monte Carlo simulation, and Bayesian inference.

**University of Twente — Data Science Module**
**Topics: Data Mining · Feature Extraction from Time Series**

---

## Key Results

| Metric | Value |
|--------|-------|
| Strategy prediction accuracy (dry, 2025) | **71% exact match** |
| Top-5 accuracy (dry, 2025) | **86%** |
| Tyre degradation model MAE | **0.079 s/lap** |
| Monte Carlo simulation speed | **~9,000 sims/sec** |

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

| Fold | Training | CV MAE | Dry Exact | Dry Top 5 |
|------|----------|--------|-----------|-----------|
| 2022 → 2023 | 967 stints | 0.105s | 40% | 50% |
| 2022-23 → 2024 | 1,982 stints | 0.087s | 52% | 71% |
| 2022-24 → 2025 | 3,006 stints | 0.079s | **71%** | **86%** |

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
