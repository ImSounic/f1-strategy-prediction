"""RL-3a: head-to-head RL main vs anchors + MC/Phase-3 plan, with bootstrap CIs.

A neutral identical-driver field isolates *strategy*: all cars use one representative
driver profile, differing only by controller. Field = 1 RL car (rotating slot) + 20
baselines split across {anchor onestop, anchor twostop, mc}. Aggregate over N seeds.

Usage:
    python -m src.rl.eval.run_beat_baselines --races 8 --circuit bahrain    # smoke
    python -m src.rl.eval.run_beat_baselines --races 200 --circuit bahrain
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import ray

from src.rl.build_env_config import build_env_config
from src.rl.eval.model_loader import load_main_module
from src.rl.eval.controllers import ScriptedController, RLController
from src.rl.eval.plans import anchor_plan, load_mc_plan
from src.rl.eval.race_runner import run_race
from src.rl.eval.metrics import win_rate, mean_finish, bootstrap_ci
from src.simulation.regulation_profiles import get_profile

BASELINES = ["anchor_onestop", "anchor_twostop", "mc"]


def _neutral_drivers(drivers):
    """All cars share a representative (median) driver so finish reflects strategy."""
    mid = drivers[len(drivers) // 2]
    return [deepcopy(mid) for _ in drivers]


def main():
    ap = argparse.ArgumentParser(description="RL-3a beat-baselines eval")
    ap.add_argument("--races", type=int, default=200)
    ap.add_argument("--circuit", type=str, default="bahrain")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--checkpoint", type=str, default="models/rl_league/league")
    ap.add_argument("--module-id", type=str, default="main_1")
    ap.add_argument("--out", type=str, default="results/rl_eval/beat_baselines.json")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    cfg = build_env_config(args.season, args.circuit)
    circuit, drivers = cfg["circuit"], _neutral_drivers(cfg["drivers"])
    profile = get_profile(args.season)
    n = len(drivers)

    module, algo = load_main_module(args.checkpoint, args.module_id)
    mc_start, mc_plan = load_mc_plan(
        f"results/scenarios_{args.circuit}_{args.season}.json", circuit.total_laps)

    def make_controller(kind):
        if kind == "rl":
            return RLController(module)
        if kind == "mc":
            return ScriptedController(mc_start, mc_plan)
        start, plan = anchor_plan(kind.replace("anchor_", ""))
        return ScriptedController(start, plan)

    finishes = {k: [] for k in ["rl", *BASELINES]}
    paired = {b: ([], []) for b in BASELINES}

    for s in range(args.races):
        rl_slot = s % n
        kinds = [None] * n
        kinds[rl_slot] = "rl"
        others = [i for i in range(n) if i != rl_slot]
        for j, i in enumerate(others):
            kinds[i] = BASELINES[j % len(BASELINES)]
        controllers = [make_controller(k) for k in kinds]
        pos = run_race(circuit, drivers, controllers, profile, seed=s)
        for i, k in enumerate(kinds):
            finishes[k].append(pos[i])
        rl_pos = pos[rl_slot]
        for b in BASELINES:
            b_slots = [i for i, k in enumerate(kinds) if k == b]
            if b_slots:
                paired[b][0].append(rl_pos)
                paired[b][1].append(pos[b_slots[0]])

    report = {"circuit": args.circuit, "season": args.season, "races": args.races,
              "module_id": args.module_id,
              "mean_finish": {k: mean_finish(v) for k, v in finishes.items()},
              "mean_finish_ci95": {k: bootstrap_ci(v) for k, v in finishes.items()},
              "rl_winrate_vs": {b: win_rate(paired[b][0], paired[b][1]) for b in BASELINES}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"✓ wrote {out}", flush=True)
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
