"""
RLlib callbacks for the RL-2b league (thin adapter over league.py)
=================================================================
on_episode_end reads each car's finish position (emitted by the env in its info
dict) and the car's policy id (episode.module_for), then records the race's
finishing order into a per-env-runner WinRateMatrix. The trainer pulls and merges
these onto the driver each iteration. Bookkeeping never crashes training.
"""
from __future__ import annotations

from ray.rllib.callbacks.callbacks import RLlibCallback

from src.rl.league import WinRateMatrix


class LeagueCallbacks(RLlibCallback):
    def on_environment_created(self, *, env_runner=None, **kwargs):
        if env_runner is not None and not hasattr(env_runner, "_winrates"):
            env_runner._winrates = WinRateMatrix()

    def on_episode_end(self, *, episode, env_runner=None, **kwargs):
        try:
            order = []
            for agent_id in episode.agent_ids:
                pid = episode.module_for(agent_id)
                info = episode.agent_episodes[agent_id].get_infos(-1)
                if isinstance(info, dict) and "finish" in info:
                    order.append((str(pid), int(info["finish"])))
            if len(order) < 2:
                return
            wr = getattr(env_runner, "_winrates", None)
            if wr is None and env_runner is not None:
                wr = env_runner._winrates = WinRateMatrix()
            if wr is not None:
                wr.record_race(order)
        except Exception:  # noqa: BLE001 — bookkeeping must never break training
            return
