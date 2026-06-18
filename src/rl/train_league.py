"""
RL-2b league trainer (full AlphaStar mechanics)
===============================================
One PPO Algorithm with multiple learning policies (main/mexp/lexp) + frozen
scripted anchors. Each iteration:
  1. round-robin a training-focus learner and assemble the 21-car field via PFSP
     (opponents drawn by role from snapshots + anchors + other learners),
  2. train,
  3. pull each env-runner's WinRateMatrix to the driver and merge it,
  4. manage the league: snapshot a dominant main, reset a successful/aged exploiter
     (guarded — a league-management API error logs and is skipped, never killing the run).

Usage:
    python -m src.rl.train_league --iters 2 --workers 2     # smoke
    python -m src.rl.train_league --iters 300 --workers 14
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from src.rl.multiagent_env import F1MultiAgentEnv
from src.rl.build_env_config import build_env_config
from src.rl.league import (
    LeagueConfig, learner_ids, role_of, opponent_pool, focus_count,
    assemble_field, pfsp_weights, league_winrate, should_snapshot,
    should_reset_exploiter, WinRateMatrix, ROLE_MAIN, ROLE_MEXP, ROLE_LEXP,
)
from src.rl.league_callbacks import LeagueCallbacks
from src.rl.scripted_anchors import ScriptedAnchorModule, ANCHOR_PLANS


class _MatrixStub:
    """Used before any race is recorded: every rate is the 0.5 prior."""

    def rate(self, a, b, prior=0.5):
        return prior


def build_assignment(cfg, focus, learners, snapshots, anchors, n_cars, matrix, rng):
    """Assemble the per-car policy grid: focus learner + PFSP-sampled opponents."""
    role = role_of(focus)
    pool = opponent_pool(role, focus, learners, snapshots, anchors)
    if not pool:
        pool = list(anchors) or list(learners)

    def sample_opponent():
        rates = [matrix.rate(focus, o) for o in pool]
        weights = pfsp_weights(rates, p=cfg.pfsp_p, eps=cfg.pfsp_eps)
        return rng.choices(pool, weights=weights, k=1)[0]

    return assemble_field(n_cars, focus, focus_count(n_cars, cfg.focus_share), sample_opponent)


def pull_winrates(algo):
    """Pull each env-runner's WinRateMatrix counts to the driver and reset them.
    Returns a list of (wins, games) dicts to merge into the driver-side matrices."""
    def _pull(runner):
        wr = getattr(runner, "_winrates", None)
        if wr is None:
            return ({}, {})
        w, g = wr.raw_counts()
        wr._w.clear(); wr._g.clear()
        return (w, g)

    try:
        parts = algo.env_runner_group.foreach_env_runner(_pull, local_env_runner=True)
    except TypeError:
        parts = algo.env_runner_group.foreach_env_runner(_pull)
    except Exception:  # noqa: BLE001
        return []
    return parts or []


def restore_learner_weights(algo, learners, path):
    """Best-effort resume: copy learner (main/mexp/lexp) weights from a prior
    checkpoint into this fresh algo (snapshot pool starts empty). Guarded so a
    failure just means we start fresh."""
    try:
        from ray.rllib.algorithms.algorithm import Algorithm
        # from_checkpoint runs the path through pyarrow's URI parser -> needs absolute.
        prev = Algorithm.from_checkpoint(str(Path(path).resolve()))
    except Exception as e:  # noqa: BLE001
        print(f"! restore failed, starting fresh: {type(e).__name__}: {e}", flush=True)
        return
    for pid in learners:
        try:
            state = prev.get_module(pid).get_state()
            algo.get_module(pid).set_state(state)
            algo.env_runner_group.foreach_env_runner(
                lambda r, p=pid, s=state: r.module[p].set_state(s))
        except Exception as e:  # noqa: BLE001
            print(f"  ! restore {pid} failed: {type(e).__name__}: {e}", flush=True)
    try:
        prev.stop()
    except Exception:  # noqa: BLE001
        pass
    print(f"✓ restored learner weights from {path}", flush=True)


def _spec_like(module, inference_only):
    """Build an RLModuleSpec matching an already-built module (class, spaces, model
    config). add_module needs an explicit module_class — a bare RLModuleSpec fails
    with 'RLModule class is not set'."""
    return RLModuleSpec(
        module_class=type(module),
        observation_space=module.observation_space,
        action_space=module.action_space,
        model_config=getattr(module, "model_config", None),
        catalog_class=getattr(module, "catalog_class", None),
        inference_only=inference_only,
    )


def _add_frozen_snapshot(algo, new_id, src_id):
    """Copy src_id's current weights into a new frozen module new_id on the runners."""
    src = algo.get_module(src_id)
    src_state = src.get_state()
    algo.add_module(module_id=new_id, module_spec=_spec_like(src, inference_only=True),
                    add_to_learners=False)
    algo.env_runner_group.foreach_env_runner(
        lambda r, nid=new_id, s=src_state: r.module[nid].set_state(s))


def manage_league(algo, driver_matrix, cfg, state, anchors, counters):
    """Snapshot dominant mains; reset successful/aged exploiters. Fully guarded."""
    learners, snaps = state["learners"], state["snapshots"]
    mains = [l for l in learners if role_of(l) == ROLE_MAIN]

    for main_id in mains:
        opp = snaps + anchors + [m for m in mains if m != main_id]
        if should_snapshot(driver_matrix, main_id, opp, counters[main_id],
                           cfg.snapshot_threshold, cfg.snapshot_every_steps):
            new_id = f"snap_{len(snaps)}"
            try:
                _add_frozen_snapshot(algo, new_id, main_id)
                snaps.append(new_id)
                counters[main_id] = 0
                print(f"  + snapshot {new_id} <- {main_id} "
                      f"(wr {league_winrate(driver_matrix, main_id, opp):.2f})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  ! snapshot {new_id} <- {main_id} failed: {type(e).__name__}: {e}", flush=True)

    for exp_id in [l for l in learners if role_of(l) in (ROLE_MEXP, ROLE_LEXP)]:
        targets = (mains if role_of(exp_id) == ROLE_MEXP
                   else snaps + anchors + [x for x in learners if x != exp_id])
        if should_reset_exploiter(driver_matrix, exp_id, targets, counters[exp_id],
                                  cfg.exploiter_threshold, cfg.exploiter_max_steps):
            snap_id = f"snap_{len(snaps)}"
            try:
                train_spec = _spec_like(algo.get_module(exp_id), inference_only=False)
                _add_frozen_snapshot(algo, snap_id, exp_id)
                snaps.append(snap_id)
                algo.remove_module(exp_id)
                algo.add_module(module_id=exp_id, module_spec=train_spec, add_to_learners=True)
                counters[exp_id] = 0
                print(f"  ~ reset {exp_id} (snapshotted {snap_id})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  ! reset {exp_id} failed: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="RL-2b league trainer")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--no-manage-league", action="store_true",
                    help="disable runtime snapshot/reset (train policies only)")
    ap.add_argument("--snapshot-threshold", type=float, default=None,
                    help="override main->snapshot win-rate threshold (set 0.0 to force in a smoke)")
    ap.add_argument("--exploiter-threshold", type=float, default=None,
                    help="override exploiter-reset win-rate threshold (set 0.0 to force in a smoke)")
    ap.add_argument("--wr-window", type=int, default=30,
                    help="iterations per rolling window for the reported recent win-rate")
    ap.add_argument("--restore", type=str, default=None,
                    help="path to a prior checkpoint; loads learner weights (fresh snapshot pool)")
    ap.add_argument("--out", type=str, default="models/rl_league/league")
    args = ap.parse_args()

    cfg = LeagueConfig()
    if args.snapshot_threshold is not None:
        cfg.snapshot_threshold = args.snapshot_threshold
    if args.exploiter_threshold is not None:
        cfg.exploiter_threshold = args.exploiter_threshold
    learners = learner_ids(cfg)
    anchors = list(ANCHOR_PLANS.keys())
    rng = random.Random(0)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    env_config = build_env_config(args.season, args.circuit)
    register_env("f1_ma", lambda c: F1MultiAgentEnv(c))
    n_cars = len(env_config["drivers"])

    driver_matrix = WinRateMatrix()
    state = {"learners": learners, "snapshots": [],
             "field": build_assignment(cfg, learners[0], learners, [], anchors,
                                        n_cars, _MatrixStub(), rng)}
    counters = {l: 0 for l in learners}

    def policy_mapping_fn(agent_id, episode=None, **kw):
        idx = int(str(agent_id).split("_")[1])
        field = state["field"]
        return field[idx % len(field)]

    module_specs = {pid: RLModuleSpec() for pid in learners}
    for name in anchors:
        module_specs[name] = RLModuleSpec(
            module_class=ScriptedAnchorModule,
            model_config={"plan": ANCHOR_PLANS[name]},
        )

    config = (
        PPOConfig()
        .environment("f1_ma", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=set(learners) | set(anchors),
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=learners,
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs))
        .env_runners(num_env_runners=args.workers)
        .callbacks(LeagueCallbacks)
        .training(train_batch_size=4000, num_epochs=args.num_epochs, gamma=0.99, lr=3e-4)
    )
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
    if args.restore:
        restore_learner_weights(algo, learners, args.restore)

    recent_matrix = WinRateMatrix()                  # rolling window for honest reporting
    mains = [l for l in learners if role_of(l) == ROLE_MAIN]

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.iters + 1):
        if (i - 1) % args.wr_window == 0:            # start a fresh reporting window
            recent_matrix = WinRateMatrix()
        focus = learners[(i - 1) % len(learners)]                    # round-robin coverage
        state["field"] = build_assignment(cfg, focus, learners, state["snapshots"],
                                           anchors, n_cars, driver_matrix, rng)
        result = algo.train()

        for w, g in pull_winrates(algo):
            if w or g:
                driver_matrix.merge_counts(w, g)     # cumulative: drives PFSP + snapshots
                recent_matrix.merge_counts(w, g)     # windowed: reporting only
        steps = int(result.get("env_runners", {}).get("num_env_steps_sampled", 4000) or 4000)
        for l in learners:
            counters[l] += steps
        if not args.no_manage_league:
            manage_league(algo, driver_matrix, cfg, state, anchors, counters)

        wr_recent = np.mean([league_winrate(recent_matrix, m, anchors) for m in mains])
        wr_life = np.mean([league_winrate(driver_matrix, m, anchors) for m in mains])
        er = result.get("env_runners", {})
        print(f"iter {i:>3} | return {er.get('episode_return_mean')} | focus {focus} "
              f"| wr {wr_recent:.3f} (life {wr_life:.3f}) | snaps {len(state['snapshots'])}",
              flush=True)
        if i % args.checkpoint_every == 0 or i == args.iters:
            print(f"  checkpoint -> {algo.save(str(out))}", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
