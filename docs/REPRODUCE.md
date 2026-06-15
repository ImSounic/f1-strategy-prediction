# Reproducing the project after a fresh clone

This guide gets a freshly cloned repo to a fully working state on **JupyterLab
(A10)** and the **HPC cluster (L40S)**. Read the "What's in git vs. what isn't"
section first — it explains why a fresh clone is *almost* complete and what (if
anything) you need to regenerate.

---

## What's in git vs. what isn't

**Committed (you already have it after cloning):**
- All source code (`src/`), configs, the frontend, the Makefile.
- **Engineered feature data** — `data/processed/clean_laps.parquet` and
  `data/features/*.parquet`. These are the inputs the models actually train on.
- Small raw data — Jolpica, OpenF1, weather, race-control, Pirelli CSV.
- All **trained models** (`models/`, incl. 24 RL agents) and **results**.

**NOT committed (gitignored — regenerate only if you need it):**
| Item | Restore with | Needed for |
|------|--------------|-----------|
| `venv/` | `make setup` | everything Python |
| `frontend/node_modules/` | `cd frontend && npm install` | the frontend |
| `data/raw/fastf1/{laps,sessions,track_status}/` (~648 MB) | `make ingest` | only a **from-raw** rebuild |
| `results/strategy_*.json` | `make precompute` | regeneratable; frontend has its own copies |

**Key point:** because feature data and models are committed, you do **not**
need the 648 MB raw lap download to work on this project. Re-ingest only if you
change feature engineering or extend the data.

> ⚠️ **Gotcha baked into the rebuild:** `models/tyre_deg_production.json` (loaded
> by the simulator, RL, sensitivity and scenario engines) is a byte-identical
> copy of `best_xgboost_model.json`. No script wrote it originally. `make model`
> now recreates it automatically via a copy step, and `make distclean` is the
> only clean target that removes it. Plain `make clean` keeps models intact.

---

## Two reproduction paths

- **Fast path (minutes)** — use committed feature data, rebuild models/results.
  This is what you want 95% of the time.
- **Full path (hours)** — re-download raw data from the APIs and rebuild
  everything from scratch. Only for verifying end-to-end reproducibility or
  changing the data pipeline.

---

## JupyterLab (A10) — your daily driver

```bash
git clone <repo-url> && cd f1-strategy-prediction

# 1. Environment (core deps; add --rl if you'll touch src/rl/* here)
make setup            # = bash scripts/setup_env.sh
source venv/bin/activate

# 2. Verify the clone is wired up correctly
python scripts/verify_setup.py     # or: make verify

# 3a. FAST PATH — rebuild models + results from committed features
make all              # model → simulate → analyze → visualize (no raw needed)

# 3b. (optional) frontend
cd frontend && npm install && npm run dev
```

`make all` skips ingestion. If you only want to confirm existing artifacts work,
`make verify` alone is enough.

### Full path — rebuild from raw (optional)
```bash
make all-from-raw     # ingest (~648 MB FastF1 download) → prepare → model → … → visualize
```
`make all-from-raw` runs ingestion + feature engineering before modeling. Use it
only to verify end-to-end reproducibility or after changing the data pipeline.

---

## HPC cluster (L40S) — batch / RL training

The HPC value is **unattended batch jobs**, not GPU power (this workload is
CPU-bound). Your cluster caps you at **2 parallel sbatch jobs**, which the RL
array script already respects.

The HPC uses **conda** (the env spec is `environment.yml`).

```bash
git clone <repo-url> && cd f1-strategy-prediction

# 1. Environment WITH the RL stack (conda)
#    (module load first if your cluster requires it)
# module load miniconda3
conda env create -f environment.yml     # creates env "f1-strategy" (full stack)
conda activate f1-strategy
python scripts/verify_setup.py
#   Update later with:  conda env update -f environment.yml --prune
#   CPU-only torch:     pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Train all 24 PPO agents as a capped array job (2 at a time)
sbatch scripts/hpc/train_rl.sbatch
squeue --me                         # watch progress
#   Override knobs:
#   SEASON=2025 TIMESTEPS=300000 sbatch scripts/hpc/train_rl.sbatch

# 3. Once training finishes, evaluate + export frontend data
python -m src.rl.evaluate --all
```

Trained models land in `models/rl/ppo_<circuit>_2025.zip`; eval comparisons in
`results/rl_comparison_*.json`. Commit + push these from wherever you ran them,
then pull on the other machine.

> The sbatch script requests CPU cores only. If a future recurrent/large policy
> actually needs the GPU, uncomment a `#SBATCH --gres=gpu:1` line and drop the
> `CUDA_VISIBLE_DEVICES=""` export.

---

## Your sync workflow (commit/push/pull by hand)

You commit and push manually, then pull on whichever machine runs the job:
- **Code / configs / small artifacts** → normal git.
- **Big regenerated artifacts** (raw laps, large model zips) → keep gitignored
  and regenerate per-machine, *or* set up DVC later (planned — Tier 3) so they
  version cleanly without bloating git. For now: regenerate, don't commit them.

---

## One-line health check

```bash
python scripts/verify_setup.py
```
Exit code 0 = the committed pipeline can run. It checks Python version, every
dependency (core + RL separately), the required committed files, and that the
production model loads.
