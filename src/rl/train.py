"""
RL Agent Training
==================
Train PPO agents to learn F1 pit stop timing.

Modes:
    --circuit bahrain    Train a specialist agent for one circuit
    --all                Train agents for all circuits
    --universal          Train one agent across all circuits (generalised)

The agent trains entirely in simulation — no real-world data needed
beyond what calibrates the environment (deg model, SC priors).

Training takes ~2-5 min per circuit on CPU (500K timesteps).

Usage:
    python -m src.rl.train --circuit bahrain --timesteps 500000
    python -m src.rl.train --all --timesteps 300000
    python -m src.rl.train --universal --timesteps 1000000
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.rl.environment import F1StrategyEnv
from src.simulation.strategy_simulator import (
    load_circuit_config,
    COMPOUND_HARDNESS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TrainingLogger(BaseCallback):
    """Custom callback to log training progress."""
    
    def __init__(self, log_interval: int = 10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
    
    def _on_step(self) -> bool:
        # Collect episode info from monitor
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                # Only record race time from completed episodes
                if "total_time" in info:
                    self.episode_times.append(info["total_time"])
        
        if self.num_timesteps % self.log_interval == 0 and self.episode_rewards:
            recent = self.episode_rewards[-100:]
            recent_times = self.episode_times[-100:] if self.episode_times else []
            
            msg = (
                f"  Step {self.num_timesteps:>8,} | "
                f"Mean reward: {np.mean(recent):>7.1f} | "
                f"Std: {np.std(recent):>5.1f}"
            )
            if recent_times:
                msg += f" | Mean race time: {np.mean(recent_times):>8.1f}s"
            
            logger.info(msg)
        
        return True


def build_env(
    circuit_key: str,
    season: int,
    config: dict,
) -> F1StrategyEnv:
    """Build environment for a specific circuit."""
    raw_paths = config["paths"]["raw"]
    fuel_config = config["modeling"]["fuel_model"]
    
    circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    sc_priors_path = Path("models/safety_car_priors.json")
    weather_dir = Path(raw_paths["fastf1"]) / "weather"
    
    circuit = load_circuit_config(circuit_key, season, circuit_csv, sc_priors_path, weather_dir)
    
    deg_model = xgb.XGBRegressor()
    deg_model.load_model("models/tyre_deg_production.json")
    
    with open("models/comparison_results.json") as f:
        comp = json.load(f)
    feature_cols = comp["experiment"]["feature_columns"]
    
    env = F1StrategyEnv(circuit, deg_model, feature_cols, fuel_config)
    return Monitor(env)


def train_agent(
    circuit_key: str,
    season: int = 2025,
    total_timesteps: int = 500_000,
    config_path: str = "configs/config.yaml",
) -> dict:
    """
    Train a PPO agent for a specific circuit.
    
    Returns training summary with performance metrics.
    """
    config = load_config(config_path)
    
    logger.info("=" * 65)
    logger.info(f"  TRAINING RL AGENT — {circuit_key.upper()}")
    logger.info("=" * 65)
    
    # Build environment
    env = build_env(circuit_key, season, config)
    eval_env = build_env(circuit_key, season, config)
    
    logger.info(f"  Circuit:    {circuit_key}")
    logger.info(f"  Timesteps:  {total_timesteps:,}")
    logger.info(f"  Obs space:  {env.observation_space.shape}")
    logger.info(f"  Action:     Discrete(2) — stay/pit")
    
    # PPO hyperparameters (tuned for F1 strategy)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,           # steps per rollout (≈35 races per rollout)
        batch_size=64,
        n_epochs=10,
        gamma=0.998,            # high discount — care about total race time
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # moderate exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto",           # Use GPU if available
        policy_kwargs={
            "net_arch": [128, 128],  # 2-layer MLP
        },
        verbose=0,
        seed=42,
    )
    
    logger.info(f"  Policy:     MLP [128, 128]")
    logger.info(f"  Algorithm:  PPO (gamma={model.gamma}, lr={model.learning_rate})")
    logger.info(f"\n  Training...")
    
    t0 = time.time()
    
    callback = TrainingLogger(log_interval=50_000)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    
    elapsed = time.time() - t0
    races_simulated = total_timesteps // env.unwrapped.total_laps
    
    logger.info(f"\n  ✓ Training complete in {elapsed:.0f}s")
    logger.info(f"    Races simulated: ~{races_simulated:,}")
    logger.info(f"    Speed: {total_timesteps / elapsed:.0f} steps/sec")
    
    # Save model
    model_dir = Path("models/rl")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"ppo_{circuit_key}_{season}"
    model.save(str(model_path))
    logger.info(f"    Saved: {model_path}")
    
    # ── Quick evaluation ──
    logger.info(f"\n  Evaluating (100 races)...")
    eval_results = evaluate_agent(model, eval_env, n_episodes=100)
    
    logger.info(f"    Median race time: {eval_results['median_time']:.1f}s")
    logger.info(f"    Mean stops:       {eval_results['mean_stops']:.2f}")
    logger.info(f"    Mean SC events:   {eval_results['mean_sc']:.2f}")
    logger.info(f"    Compound variety: {eval_results['mean_compounds']:.1f} types")
    logger.info(f"    Legal races:      {eval_results['legal_pct']:.0f}%")
    
    env.close()
    eval_env.close()
    
    return {
        "circuit_key": circuit_key,
        "season": season,
        "total_timesteps": total_timesteps,
        "training_time_s": round(elapsed, 1),
        "races_simulated": races_simulated,
        "model_path": str(model_path),
        **eval_results,
    }


def evaluate_agent(model, env, n_episodes: int = 100) -> dict:
    """Run deterministic evaluation episodes."""
    times = []
    stops = []
    sc_events = []
    compounds = []
    legal = []
    
    for i in range(n_episodes):
        obs, info = env.reset(seed=1000 + i)
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        times.append(info["total_time"])
        stops.append(info["stops_done"])
        sc_events.append(info["sc_events"])
        compounds.append(len(info["compounds_used"]))
        legal.append(len(info["compounds_used"]) >= 2)
    
    return {
        "median_time": float(np.median(times)),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "p5_time": float(np.percentile(times, 5)),
        "p95_time": float(np.percentile(times, 95)),
        "mean_stops": float(np.mean(stops)),
        "mean_sc": float(np.mean(sc_events)),
        "mean_compounds": float(np.mean(compounds)),
        "legal_pct": float(100 * np.mean(legal)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for F1 strategy")
    parser.add_argument("--circuit", type=str, help="Train for specific circuit")
    parser.add_argument("--all", action="store_true", help="Train for all circuits")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    if args.all:
        config = load_config(args.config)
        raw_paths = config["paths"]["raw"]
        circuit_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
        circuits_df = pd.read_csv(circuit_csv)
        circuit_keys = sorted(
            circuits_df[circuits_df["season"] == args.season]["circuit_key"].unique()
        )
        
        logger.info(f"Training RL agents for {len(circuit_keys)} circuits")
        
        all_results = []
        for key in circuit_keys:
            try:
                result = train_agent(key, args.season, args.timesteps, args.config)
                all_results.append(result)
            except Exception as e:
                logger.error(f"  ✗ {key}: {e}")
        
        # Save summary
        summary_path = Path("models/rl/training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✓ Summary saved: {summary_path}")
    
    elif args.circuit:
        train_agent(args.circuit, args.season, args.timesteps, args.config)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()