"""Pure evaluation statistics (win-rate, mean finish, bootstrap CIs)."""
from __future__ import annotations

import numpy as np


def win_rate(a_finishes, b_finishes) -> float:
    """Fraction of paired races where a finished ahead of b (lower pos = better).
    Ties are not wins. Lists must be equal length and non-empty."""
    a, b = list(a_finishes), list(b_finishes)
    assert len(a) == len(b) and a, "need equal-length non-empty paired finishes"
    wins = sum(1 for x, y in zip(a, b) if x < y)
    return wins / len(a)


def mean_finish(finishes) -> float:
    vals = list(finishes)
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean. Deterministic given seed."""
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=vals.size, replace=True).mean()
                      for _ in range(n)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)
