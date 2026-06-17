"""
Multi-agent F1 strategy environment (RLlib)
===========================================
Every car is an agent sharing one driver-conditioned policy. Each step advances
the validated multi_car_sim by one lap, applying each agent's pit/compound action.
Reward is terminal (positions gained). Built on the step-able MultiCarRaceSim so
the agents train in the SAME physics validated in Phase 2/3.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.simulation.multi_car_sim import MultiCarRaceSim, Strategy
from src.simulation.regulation_profiles import get_profile
from src.rl.ma_obs import (
    action_to_compound, legal_action_mask, terminal_reward, build_obs, OBS_DIM,
)

_COMPOUND_IDX = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}


class F1MultiAgentEnv(MultiAgentEnv):
    """N cars, shared policy. Agent ids are 'car_0' ... 'car_{N-1}'."""

    def __init__(self, config: dict):
        super().__init__()
        self.circuit = config["circuit"]
        self.drivers = config["drivers"]            # list[DriverConfig], grid order
        self.season = config.get("season", 2025)
        self.profile = get_profile(self.season)
        self.n_cars = len(self.drivers)
        self.agents = [f"car_{i}" for i in range(self.n_cars)]
        self.possible_agents = list(self.agents)
        self.grid = {f"car_{i}": i + 1 for i in range(self.n_cars)}  # grid pos (1-indexed)

        self.observation_space = spaces.Box(0.0, 1.5, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        # New-API-stack MultiAgentEnv wants per-agent space dicts (all agents share).
        self.observation_spaces = {a: self.observation_space for a in self.agents}
        self.action_spaces = {a: self.action_space for a in self.agents}
        self._default_strat = Strategy(stints=[("MEDIUM", self.circuit.total_laps)], name="x")

    def _obs_for(self, i: int) -> np.ndarray:
        car = self.sim.cars[i]
        driver = self.drivers[i]
        gaps = self.sim._compute_gaps(self.sim.cars, self.sim.positions)
        order = list(np.argsort(self.sim.positions))
        pos_idx = order.index(i)
        gap_behind = gaps[order[pos_idx + 1]] if pos_idx + 1 < self.n_cars else 999.0
        state = dict(
            lap=self.sim.lap, total_laps=self.circuit.total_laps,
            tyre_age=car.tyre_age, compound_idx=_COMPOUND_IDX.get(car.tyre_compound, 1),
            cumulative_deg=0.0, position=int(self.sim.positions[i]), n_cars=self.n_cars,
            gap_ahead=float(gaps[i]), gap_behind=float(gap_behind),
            fuel_frac=max(0.0, 1.0 - self.sim.burn_rate * self.sim.lap / max(self.profile.start_fuel_kg, 1)),
            sc_active=int(self.sim.sc_active or self.sim.vsc_active),
            stops_done=car.stops_done, max_stops=3, compounds_used=len(car.compounds_used),
            laps_since_sc=0,
            driver_pace=min(driver.pace_delta, 2.0) / 2.0,
            driver_overtaking=driver.overtaking, driver_tyre=driver.tyre_management,
            pit_loss=self.circuit.pit_loss_seconds, sc_prob=self.circuit.sc_prob_per_race,
            overtaking_difficulty=self.circuit.overtaking_difficulty,
        )
        return build_obs(state)

    def reset(self, *, seed=None, options=None):
        strategies = [self._default_strat] * self.n_cars
        self.sim = MultiCarRaceSim(self.circuit, self.drivers, strategies, 0,
                                   self._default_strat, greedy_sc=False, profile=self.profile)
        self.sim.reset(seed=seed if seed is not None else 0)
        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        return obs, {a: {} for a in self.agents}

    def step(self, action_dict: dict):
        pit_override = {}
        for i, a in enumerate(self.agents):
            act = int(action_dict.get(a, 0))
            car = self.sim.cars[i]
            mask = legal_action_mask(car.stops_done, car.tyre_age, self.sim.lap + 1,
                                     self.circuit.total_laps)
            comp = action_to_compound(act) if mask[act] else None
            pit_override[i] = comp
        self.sim.step(pit_override)

        done = self.sim.done
        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        if not done:
            rewards = {a: 0.0 for a in self.agents}
        else:
            rewards = {}
            for i, a in enumerate(self.agents):
                finish = int(self.sim.positions[i])
                # Dry race on a single compound = disqualification (terminal_reward
                # returns the DSQ penalty, strictly worse than any legal finish).
                legal = len(self.sim.cars[i].compounds_used) >= 2
                rewards[a] = terminal_reward(self.grid[a], finish, used_two_compounds=legal)
        terminateds = {a: done for a in self.agents}
        terminateds["__all__"] = done
        truncateds = {a: False for a in self.agents}
        truncateds["__all__"] = False
        # On the final lap, surface each car's finish position so the league
        # callback can record pairwise win-rates (it can't reach the sim directly).
        if done:
            infos = {a: {"finish": int(self.sim.positions[i])} for i, a in enumerate(self.agents)}
        else:
            infos = {a: {} for a in self.agents}
        return obs, rewards, terminateds, truncateds, infos
