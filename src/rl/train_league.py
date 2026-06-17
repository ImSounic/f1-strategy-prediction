"""
RL-2b league trainer
====================
One PPO Algorithm with multiple learning policies (main/mexp/lexp) + frozen
scripted anchors; a league-aware policy_mapping_fn assigns each car a policy per
iteration via PFSP; LeagueCallbacks records win-rates. Snapshot-adding / exploiter
resets use the pure league.py predicates (wired in once the multi-policy build is
confirmed stable on HPC).

Usage:
    python -m src.rl.train_league --iters 2 --workers 2     # smoke
    python -m src.rl.train_league --iters 300 --workers 14
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from src.rl.multiagent_env import F1MultiAgentEnv
from src.rl.build_env_config import build_env_config
from src.rl.league import (
    LeagueConfig, learner_ids, role_of, opponent_pool, focus_count,
    assemble_field, pfsp_weights,
)
from src.rl.league_callbacks import LeagueCallbacks
from src.rl.scripted_anchors import ScriptedAnchorModule, ANCHOR_PLANS


class _MatrixStub:
    """Used before any race is recorded: every rate is the 0.5 prior."""

    def rate(self, a, b, prior=0.5):
        return prior


def build_assignment(cfg, learners, anchors, n_cars, matrix, rng):
    """Pick a focus learner and assemble the per-car policy grid via PFSP."""
    focus = rng.choice(learners)
    role = role_of(focus)
    pool = opponent_pool(role, focus, learners, snapshots=[], anchors=anchors)
    if not pool:
        pool = list(anchors) or list(learners)

    def sample_opponent():
        rates = [matrix.rate(focus, o) for o in pool]
        weights = pfsp_weights(rates, p=cfg.pfsp_p, eps=cfg.pfsp_eps)
        return rng.choices(pool, weights=weights, k=1)[0]

    n_focus = focus_count(n_cars, cfg.focus_share)
    return assemble_field(n_cars, focus, n_focus, sample_opponent)


def main():
    ap = argparse.ArgumentParser(description="RL-2b league trainer")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--out", type=str, default="models/rl_league/league")
    args = ap.parse_args()

    cfg = LeagueConfig()
    learners = learner_ids(cfg)
    anchors = list(ANCHOR_PLANS.keys())
    rng = random.Random(0)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    env_config = build_env_config(args.season, args.circuit)
    register_env("f1_ma", lambda c: F1MultiAgentEnv(c))
    n_cars = len(env_config["drivers"])

    # Shared assignment state, regenerated per iteration; read by the mapping fn.
    state = {"field": build_assignment(cfg, learners, anchors, n_cars, _MatrixStub(), rng)}

    def policy_mapping_fn(agent_id, episode=None, **kw):
        idx = int(str(agent_id).split("_")[1])
        field = state["field"]
        return field[idx % len(field)]

    module_specs = {pid: RLModuleSpec() for pid in learners}
    for name in anchors:
        # inference_only=True keeps anchors on the env-runners (they still race) but
        # excludes them from the learner — they have no trainable params to optimize.
        module_specs[name] = RLModuleSpec(
            module_class=ScriptedAnchorModule,
            model_config={"plan": ANCHOR_PLANS[name]},
            inference_only=True,
        )

    config = (
        PPOConfig()
        .environment("f1_ma", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=set(learners) | set(anchors),
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=learners,           # anchors frozen (not trained)
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs))
        .env_runners(num_env_runners=args.workers)
        .callbacks(LeagueCallbacks)
        .training(train_batch_size=4000, num_epochs=args.num_epochs, gamma=0.99, lr=3e-4)
    )
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.iters + 1):
        matrix = getattr(algo, "_winrates", None) or _MatrixStub()
        state["field"] = build_assignment(cfg, learners, anchors, n_cars, matrix, rng)
        result = algo.train()
        er = result.get("env_runners", {})
        print(f"iter {i:>3} | return {er.get('episode_return_mean')} "
              f"| focus {state['field'][0]}", flush=True)
        if i % args.checkpoint_every == 0 or i == args.iters:
            print(f"  checkpoint -> {algo.save(str(out))}", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
