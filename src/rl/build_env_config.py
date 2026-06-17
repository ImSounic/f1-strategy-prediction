"""
Env-config + policy mapping for RLlib self-play training (RL-2a)
===============================================================
build_env_config() bakes per-compound degradation into CircuitParams ONCE, so the
returned config is fully picklable and Ray rollout workers run the validated
pure-Python sim with no XGBoost at runtime. Heavy imports are lazy so this module
(and policy_mapping_fn) imports cleanly on any machine.
"""
from __future__ import annotations


def policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """Self-play: every car shares the single 'main' policy."""
    return "main"


def build_env_config(season: int = 2025, circuit_key: str = "bahrain",
                     config_path: str = "configs/config.yaml") -> dict:
    """Build a picklable env_config: {circuit, drivers, season}."""
    import json
    import yaml
    import xgboost as xgb
    from src.simulation.precompute_scenarios import load_drivers, load_circuit_as_params

    cfg = yaml.safe_load(open(config_path))
    drivers, _teams, overtaking = load_drivers(f"configs/drivers_{season}.json")
    deg = xgb.XGBRegressor()
    deg.load_model("models/tyre_deg_production.json")
    feature_cols = json.load(open("models/comparison_results.json"))["experiment"]["feature_columns"]
    circuit = load_circuit_as_params(circuit_key, season, cfg, overtaking, deg, feature_cols)
    return {"circuit": circuit, "drivers": drivers, "season": season}
