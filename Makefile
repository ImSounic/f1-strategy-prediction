.PHONY: all setup setup-rl ingest prepare model simulate analyze visualize \
        rl-train rl-eval precompute verify clean distclean

# ════════════════════════════════════════════════════════════════════
#  F1 Race Strategy Optimizer — Build Pipeline
# ════════════════════════════════════════════════════════════════════
# Quick start on a fresh clone (see docs/REPRODUCE.md):
#   make setup      — create venv + install core deps
#   make verify     — confirm the clone has everything wired up
#   make all        — rebuild models/results from committed feature data
#
# Heavy / optional:
#   make ingest     — re-download raw data from APIs (~648MB, slow)
#   make setup-rl   — add the reinforcement-learning stack (torch, SB3)
#   make rl-train   — train the 24 PPO agents (run on HPC; see sbatch)

PYTHON = python -m

# Full pipeline from committed feature parquets (NO re-ingest, NO RL).
all: prepare model simulate analyze visualize
	@echo "✓ Full pipeline complete"

# ── Environment ──────────────────────────────────────────────────────
setup:
	bash scripts/setup_env.sh

setup-rl:
	bash scripts/setup_env.sh --rl

# ── Phase 1: Data ingestion (downloads from FastF1/Jolpica/OpenF1) ────
# Regenerates the gitignored raw lap/session/track-status parquets.
# Only needed for a true from-raw rebuild — committed feature data
# already lets you skip straight to `make prepare` or `make model`.
ingest:
	$(PYTHON) src.ingestion.fastf1_extractor
	$(PYTHON) src.ingestion.jolpica_client
	$(PYTHON) src.ingestion.openf1_client
	@echo "✓ Phase 1: ingestion complete"

# ── Phase 2: Data preparation (needs raw laps from `make ingest`) ─────
prepare:
	$(PYTHON) src.preparation.clean_laps
	$(PYTHON) src.preparation.feature_engineering
	@echo "✓ Phase 2: feature engineering complete"

# ── Phase 3: Modeling ────────────────────────────────────────────────
# model_comparison writes best_xgboost_model.json; the simulator, RL,
# sensitivity and scenario engines all load models/tyre_deg_production.json,
# which is a byte-identical promotion of that file — so we copy it here.
model:
	$(PYTHON) src.modeling.model_comparison
	cp models/best_xgboost_model.json models/tyre_deg_production.json
	$(PYTHON) src.modeling.tyre_degradation_v3
	$(PYTHON) src.modeling.train_curvature
	$(PYTHON) src.modeling.safety_car_model
	$(PYTHON) src.modeling.circuit_clustering
	@echo "✓ Phase 3: modeling complete (production model promoted)"

# ── Phase 4: Simulation ──────────────────────────────────────────────
simulate:
	$(PYTHON) src.simulation.strategy_simulator --circuit bahrain --season 2024 --n-sims 1000
	$(PYTHON) src.simulation.strategy_simulator --circuit monaco --season 2024 --n-sims 1000
	@echo "✓ Phase 4: simulation complete"

# ── Phase 5a: Analysis ───────────────────────────────────────────────
analyze:
	$(PYTHON) src.analysis.shap_analysis
	$(PYTHON) src.analysis.dtw_similarity
	$(PYTHON) src.analysis.strategy_validation_rolling
	@echo "✓ Phase 5a: analysis complete"

# ── Phase 5b: Report figures ─────────────────────────────────────────
visualize:
	$(PYTHON) src.visualization.report_figures
	@echo "✓ Phase 5b: figures complete"

# ── Reinforcement learning (CPU-bound; run on HPC, see sbatch) ────────
rl-train:
	$(PYTHON) src.rl.train --all --timesteps 500000
	@echo "✓ RL training complete"

rl-eval:
	$(PYTHON) src.rl.evaluate --all
	@echo "✓ RL evaluation complete"

# ── Precompute frontend artifacts (strategies + scenarios) ───────────
precompute:
	$(PYTHON) src.scripts.precompute_all_strategies
	$(PYTHON) src.scripts.precompute_scenarios
	@echo "✓ Precompute complete"

# ── Verify a fresh clone is correctly set up ─────────────────────────
verify:
	python scripts/verify_setup.py

# ── Cleaning ─────────────────────────────────────────────────────────
# clean: remove regeneratable outputs but KEEP trained models + raw data.
clean:
	rm -f data/features/*.parquet results/figures/*.png
	@echo "✓ Cleaned regeneratable outputs (models & raw kept)"

# distclean: also remove trained models. Run `make model` to rebuild
# (this recreates tyre_deg_production.json via the promotion step above).
distclean: clean
	rm -f models/*.json models/*.pkl
	@echo "✓ Removed trained models — run 'make model' to rebuild"
