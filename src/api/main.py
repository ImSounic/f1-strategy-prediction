"""
F1 Race Strategy Optimizer — REST API
=======================================
FastAPI backend for the strategy optimizer.

Endpoints:
    GET  /                     → Health check
    GET  /circuits/{season}    → Available circuits for a season
    GET  /circuit/{key}/{season} → Circuit details + characteristics
    POST /simulate             → Run Monte Carlo strategy simulation
    GET  /validation           → Model validation results

Usage:
    uvicorn src.api.main:app --reload
    
Production:
    uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App ──
app = FastAPI(
    title="F1 Race Strategy Optimizer",
    description="Monte Carlo simulation for optimal F1 pit stop strategies",
    version="1.0.0",
)

# CORS — allow all origins for now (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load resources at startup ──
class AppState:
    """Singleton to hold loaded models and data."""
    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        self.config = yaml.safe_load(open("configs/config.yaml"))

        self.circuits_df = pd.read_csv(
            Path(self.config["paths"]["raw"]["supplementary"])
            / "pirelli_circuit_characteristics.csv"
        )

        with open("models/safety_car_priors.json") as f:
            self.sc_priors = json.load(f)

        with open("models/comparison_results.json") as f:
            comp = json.load(f)
        self.feature_cols = comp["experiment"]["feature_columns"]

        self.model = xgb.XGBRegressor()
        self.model.load_model("models/tyre_deg_production.json")

        with open("results/validation_rolling_report.json") as f:
            self.validation = json.load(f)

        self.loaded = True


state = AppState()


@app.on_event("startup")
async def startup():
    state.load()


# ── Request / Response models ──
class WeatherOverride(BaseModel):
    track_temp: Optional[float] = None
    air_temp: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None


class SimulationRequest(BaseModel):
    circuit_key: str = Field(..., description="Circuit identifier (e.g. 'bahrain', 'monaco')")
    season: int = Field(default=2025, description="Season year (2022-2025)")
    n_sims: int = Field(default=1000, ge=50, le=5000, description="Simulations per strategy")
    weather_override: Optional[WeatherOverride] = None


# ── Endpoints ──

@app.get("/")
async def health():
    return {
        "status": "healthy",
        "model": "XGBoost tyre degradation",
        "circuits": len(state.circuits_df["circuit_key"].unique()),
        "validation_accuracy": "71% exact match (2025 dry races)",
    }


@app.get("/circuits/{season}")
async def get_circuits(season: int):
    """List available circuits for a season."""
    df = state.circuits_df[state.circuits_df["season"] == season]
    if df.empty:
        raise HTTPException(404, f"No data for season {season}")

    circuits = []
    for _, row in df.sort_values("round_number").iterrows():
        sc = state.sc_priors.get(row["circuit_key"], {})
        circuits.append({
            "circuit_key": row["circuit_key"],
            "circuit_name": row["circuit_name"],
            "round_number": int(row["round_number"]),
            "total_laps": int(row["total_laps"]),
            "pit_loss_seconds": float(row["pit_loss_seconds"]),
            "compounds": f"{row['hard_compound']}/{row['medium_compound']}/{row['soft_compound']}",
            "sc_probability": round(sc.get("bayesian_sc_prob", 0.55), 3),
        })

    return {"season": season, "circuits": circuits}


@app.get("/circuit/{circuit_key}/{season}")
async def get_circuit_detail(circuit_key: str, season: int):
    """Get detailed circuit information."""
    df = state.circuits_df
    row = df[(df["circuit_key"] == circuit_key) & (df["season"] == season)]

    if row.empty:
        raise HTTPException(404, f"Circuit {circuit_key} not found for {season}")

    row = row.iloc[0]
    sc = state.sc_priors.get(circuit_key, {})

    char_cols = [
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution",
    ]

    return {
        "circuit_key": circuit_key,
        "circuit_name": row["circuit_name"],
        "total_laps": int(row["total_laps"]),
        "pit_loss_seconds": float(row["pit_loss_seconds"]),
        "compounds": f"{row['hard_compound']}/{row['medium_compound']}/{row['soft_compound']}",
        "sc_probability": round(sc.get("bayesian_sc_prob", 0.55), 3),
        "characteristics": {col: float(row[col]) for col in char_cols},
    }


@app.post("/simulate")
async def simulate(req: SimulationRequest):
    """Run Monte Carlo strategy simulation."""
    from src.simulation.strategy_simulator import (
        load_circuit_config,
        generate_strategies,
        run_monte_carlo,
    )

    t0 = time.time()

    try:
        circuit_csv = (
            Path(state.config["paths"]["raw"]["supplementary"])
            / "pirelli_circuit_characteristics.csv"
        )
        sc_path = Path("models/safety_car_priors.json")
        weather_dir = Path(state.config["paths"]["raw"]["fastf1"]) / "weather"
        fuel_config = state.config["modeling"]["fuel_model"]

        circuit = load_circuit_config(
            req.circuit_key, req.season, circuit_csv, sc_path, weather_dir,
        )

        # Apply weather overrides
        if req.weather_override:
            if req.weather_override.track_temp is not None:
                circuit.mean_track_temp = req.weather_override.track_temp
            if req.weather_override.air_temp is not None:
                circuit.mean_air_temp = req.weather_override.air_temp
            if req.weather_override.humidity is not None:
                circuit.mean_humidity = req.weather_override.humidity
            if req.weather_override.wind_speed is not None:
                circuit.mean_wind_speed = req.weather_override.wind_speed

        strategies = generate_strategies(circuit)

        all_results = []
        for i, strategy in enumerate(strategies):
            result = run_monte_carlo(
                strategy, circuit, state.model, state.feature_cols,
                fuel_config, n_sims=req.n_sims, seed=42 + i,
            )
            all_results.append(result)

        all_results.sort(key=lambda x: x["median_time"])
        elapsed = time.time() - t0

        best_median = all_results[0]["median_time"]
        top15 = all_results[:15]

        return {
            "circuit_key": req.circuit_key,
            "circuit_name": circuit.circuit_name if hasattr(circuit, 'circuit_name') else req.circuit_key,
            "season": req.season,
            "n_sims": req.n_sims,
            "n_strategies": len(all_results),
            "elapsed_seconds": round(elapsed, 2),
            "rankings": [
                {
                    "strategy_name": r["strategy_name"],
                    "compound_sequence": r.get("compound_sequence", ""),
                    "num_stops": r["num_stops"],
                    "median_time": round(r["median_time"], 1),
                    "mean_time": round(r["mean_time"], 1),
                    "std_time": round(r["std_time"], 1),
                    "p5_time": round(r["p5_time"], 1),
                    "p95_time": round(r["p95_time"], 1),
                    "mean_sc_events": round(r.get("mean_sc_events", 0), 2),
                }
                for r in top15
            ],
        }

    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {str(e)}")


@app.get("/validation")
async def get_validation():
    """Return model validation results."""
    return state.validation
