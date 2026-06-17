"""
Pure league core for the AlphaStar-style RL-2b training
=======================================================
No Ray / no torch imports — unit-testable on any machine. The RLlib callbacks
(league_callbacks.py) and trainer (train_league.py) compose these into a live
multi-policy league. Everything here is deterministic given its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

ROLE_MAIN, ROLE_MEXP, ROLE_LEXP = "main", "mexp", "lexp"


@dataclass
class LeagueConfig:
    n_main: int = 2
    n_mexp: int = 2
    n_lexp: int = 2
    focus_share: float = 0.5
    pfsp_p: float = 2.0
    pfsp_eps: float = 0.1
    snapshot_threshold: float = 0.70
    snapshot_every_steps: int = 2_000_000
    exploiter_threshold: float = 0.70
    exploiter_max_steps: int = 4_000_000


def role_of(policy_id: str) -> str:
    """'main_0' -> 'main', 'snap_3' -> 'snap', 'anchor_onestop' -> 'anchor'."""
    return policy_id.split("_")[0]


def learner_ids(cfg: LeagueConfig) -> list:
    """The actively-training policy ids (in policies_to_train)."""
    return ([f"main_{i}" for i in range(cfg.n_main)]
            + [f"mexp_{i}" for i in range(cfg.n_mexp)]
            + [f"lexp_{i}" for i in range(cfg.n_lexp)])


class WinRateMatrix:
    """Running pairwise win-rates. rate(A,B) = P(an A-car finishes ahead of a B-car)."""

    def __init__(self):
        self._w: dict = {}   # (A,B) -> A-ahead count
        self._g: dict = {}   # (A,B) -> games played

    def record_race(self, finishing: list) -> None:
        """finishing: list of (policy_id, finish_position) for every car in the race.

        Updates every co-present ordered pair. Finish positions are unique ranks, so
        there are no ties to resolve.
        """
        n = len(finishing)
        for i in range(n):
            a, pa = finishing[i]
            for j in range(n):
                if i == j:
                    continue
                b, pb = finishing[j]
                if a == b:
                    continue
                self._g[(a, b)] = self._g.get((a, b), 0) + 1
                if pa < pb:
                    self._w[(a, b)] = self._w.get((a, b), 0) + 1

    def rate(self, a: str, b: str, prior: float = 0.5) -> float:
        g = self._g.get((a, b), 0)
        return prior if g == 0 else self._w.get((a, b), 0) / g

    def games(self, a: str, b: str) -> int:
        return self._g.get((a, b), 0)

    def merge_counts(self, wins: dict, games: dict) -> None:
        """Add another matrix's raw (wins, games) counts into this one. Used on the
        driver to aggregate per-env-runner matrices each iteration."""
        for k, v in games.items():
            self._g[k] = self._g.get(k, 0) + v
        for k, v in wins.items():
            self._w[k] = self._w.get(k, 0) + v

    def raw_counts(self) -> tuple:
        """(wins, games) dict copies — for pulling a runner's matrix to the driver."""
        return dict(self._w), dict(self._g)


def pfsp_weights(rates: list, p: float = 2.0, eps: float = 0.1) -> list:
    """Prioritized fictitious self-play weights. Lower win-rate (harder opponent)
    -> higher weight. Mixed with uniform (eps) so every opponent stays reachable."""
    n = len(rates)
    if n == 0:
        return []
    hard = [max(0.0, 1.0 - r) ** p for r in rates]
    s = sum(hard)
    base = [h / s for h in hard] if s > 0 else [1.0 / n] * n
    return [(1.0 - eps) * b + eps * (1.0 / n) for b in base]


def opponent_pool(role: str, focus_id: str, learners: list,
                  snapshots: list, anchors: list) -> list:
    """Candidate opponents for a focus policy, by AlphaStar role."""
    if role == ROLE_MEXP:                                   # main-exploiter: hunt mains
        return [x for x in learners if role_of(x) == ROLE_MAIN]
    if role == ROLE_MAIN:                                   # main: pool + other mains
        other_mains = [x for x in learners if role_of(x) == ROLE_MAIN and x != focus_id]
        return list(snapshots) + list(anchors) + other_mains
    return list(snapshots) + list(anchors) + [x for x in learners if x != focus_id]  # lexp


def focus_count(n_cars: int, focus_share: float) -> int:
    return max(1, min(n_cars, round(n_cars * focus_share)))


def assemble_field(n_cars: int, focus_id: str, n_focus: int, sample_opponent) -> list:
    """Grid of policy ids: n_focus cars are the focus policy, the rest sampled opponents."""
    field = [focus_id] * min(n_focus, n_cars)
    while len(field) < n_cars:
        field.append(sample_opponent())
    return field


def league_winrate(matrix: WinRateMatrix, agent: str, opponents: list,
                   prior: float = 0.5) -> float:
    """Mean win-rate of `agent` against a set of opponents."""
    if not opponents:
        return prior
    return sum(matrix.rate(agent, o, prior) for o in opponents) / len(opponents)


def should_snapshot(matrix: WinRateMatrix, main_id: str, league_opponents: list,
                    steps_since: int, threshold: float, every_steps: int) -> bool:
    """Freeze a copy of a main agent when it dominates the league or on a step cadence."""
    return (league_winrate(matrix, main_id, league_opponents) >= threshold
            or steps_since >= every_steps)


def should_reset_exploiter(matrix: WinRateMatrix, exp_id: str, targets: list,
                           steps_alive: int, threshold: float, max_steps: int) -> bool:
    """Reset an exploiter once it beats its target enough, or after a max lifetime."""
    return (league_winrate(matrix, exp_id, targets) >= threshold
            or steps_alive >= max_steps)
