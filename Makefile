.PHONY: all setup ingest prepare model simulate analyze visualize demo api clean

# ════════════════════════════════════════════════
#  F1 Race Strategy Optimizer — Build Pipeline
# ════════════════════════════════════════════════
# Usage:
#   make setup     — Install dependencies
#   make all       — Run full pipeline (~15 min)
#   make demo      — Launch Streamlit app
#   make api       — Launch FastAPI server
#   make clean     — Remove generated outputs

PYTHON = python -m

# Full pipeline
all: ingest prepare model simulate analyze visualize
	@echo "✓ Full pipeline complete"

# Setup
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✓ Environment ready"

# Phase 1: Data Ingestion
ingest:
	$(PYTHON) src.ingestion.extract_fastf1
	$(PYTHON) src.ingestion.extract_jolpica
	$(PYTHON) src.ingestion.extract_openf1
	@echo "✓ Phase 1: Data ingestion complete"

# Phase 2: Data Preparation
prepare:
	$(PYTHON) src.preparation.clean_laps
	$(PYTHON) src.preparation.feature_engineering
	@echo "✓ Phase 2: Feature engineering complete"

# Phase 3: Modeling
model:
	$(PYTHON) src.modeling.model_comparison
	$(PYTHON) src.modeling.safety_car_model
	$(PYTHON) src.modeling.circuit_clustering
	@echo "✓ Phase 3: Modeling complete"

# Phase 4: Simulation
simulate:
	$(PYTHON) src.simulation.strategy_simulator --circuit bahrain --season 2024 --n-sims 1000
	$(PYTHON) src.simulation.strategy_simulator --circuit monaco --season 2024 --n-sims 1000
	@echo "✓ Phase 4: Simulation complete"

# Phase 5a: Analysis
analyze:
	$(PYTHON) src.analysis.shap_analysis
	$(PYTHON) src.analysis.dtw_similarity
	$(PYTHON) src.analysis.strategy_validation_rolling
	@echo "✓ Phase 5a: Analysis complete"

# Phase 5b: Visualization
visualize:
	$(PYTHON) src.visualization.report_figures
	@echo "✓ Phase 5b: Figures generated in results/figures/"

# Interactive demo
demo:
	streamlit run src/demo/app.py

# API server
api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Clean outputs (keeps raw data)
clean:
	rm -rf data/features/*.parquet
	rm -rf models/*.json models/*.pkl
	rm -rf results/
	@echo "✓ Cleaned generated outputs"
