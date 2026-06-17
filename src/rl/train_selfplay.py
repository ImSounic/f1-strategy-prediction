"""
RLlib PPO self-play training (RL-2a)
====================================
One shared 'main' policy controls all cars in F1MultiAgentEnv, single circuit.
Logs episode_return_mean per iteration and checkpoints. No league yet (RL-2b).

Usage:
    python -m src.rl.train_selfplay --iters 50 --workers 6
    python -m src.rl.train_selfplay --iters 2 --workers 2   # smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from src.rl.multiagent_env import F1MultiAgentEnv
from src.rl.build_env_config import build_env_config, policy_mapping_fn


def _episode_return(result: dict):
    """Robustly pull mean episode return across RLlib versions."""
    er = result.get("env_runners", {})
    for key in ("episode_return_mean", "episode_reward_mean"):
        if key in er:
            return er[key]
        if key in result:
            return result[key]
    return None


def main():
    ap = argparse.ArgumentParser(description="RL-2a PPO self-play training")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="models/rl_league/selfplay")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env("f1_ma", lambda cfg: F1MultiAgentEnv(cfg))
    env_config = build_env_config(args.season, args.circuit)

    config = (
        PPOConfig()
        .environment("f1_ma", env_config=env_config)
        .framework("torch")
        .multi_agent(policies={"main"}, policy_mapping_fn=policy_mapping_fn)
        .env_runners(num_env_runners=args.workers)
        .training(train_batch_size=4000, gamma=0.99, lr=3e-4)
    )
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.iters + 1):
        result = algo.train()
        print(f"iter {i:>3} | episode_return_mean {_episode_return(result)}", flush=True)
        if i % args.checkpoint_every == 0 or i == args.iters:
            ckpt = algo.save(str(out))
            print(f"  checkpoint -> {ckpt}", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
