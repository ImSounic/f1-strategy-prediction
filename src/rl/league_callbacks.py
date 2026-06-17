"""
RLlib callbacks for the RL-2b league (thin adapter over league.py)
=================================================================
on_episode_end records the race's finishing order into a per-env-runner
WinRateMatrix, keyed by each car's policy id. Snapshot/reset decisions live in
train_league.py using the pure league.py predicates. Bookkeeping never crashes
training — any error is swallowed.
"""
from __future__ import annotations

from ray.rllib.callbacks.callbacks import RLlibCallback

from src.rl.league import WinRateMatrix


class LeagueCallbacks(RLlibCallback):
    def on_environment_created(self, *, env_runner=None, **kwargs):
        if env_runner is not None and not hasattr(env_runner, "_winrates"):
            env_runner._winrates = WinRateMatrix()

    def on_episode_end(self, *, episode, env_runner=None, env=None, **kwargs):
        try:
            base = getattr(env, "unwrapped", env)
            sim = getattr(base, "sim", None)
            if sim is None:
                return
            order = []
            for i, agent_id in enumerate(base.agents):
                if hasattr(episode, "module_for"):
                    pid = episode.module_for(agent_id)
                else:
                    pid = "main_0"
                order.append((str(pid), int(sim.positions[i])))
            wr = getattr(env_runner, "_winrates", None)
            if wr is None and env_runner is not None:
                wr = env_runner._winrates = WinRateMatrix()
            if wr is not None:
                wr.record_race(order)
        except Exception:  # noqa: BLE001 — bookkeeping must never break training
            return
