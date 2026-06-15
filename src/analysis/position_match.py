"""
Position-accuracy scoring (pure)
================================
Spearman rank correlation + mean absolute position error, over a common set of
drivers. No numpy/scipy so it is unit-testable anywhere.
"""
from __future__ import annotations


def _mean(xs):
    return sum(xs) / len(xs)


def _ranks(values):
    """Fractional ranks (1-indexed), averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def spearman(x, y):
    """Spearman rank correlation between two equal-length sequences."""
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_ranks(x), _ranks(y))


def score_positions(predicted: dict, actual: dict) -> dict:
    """Score predicted vs actual finishing positions over common drivers.

    predicted/actual: {driver_code: position (1-indexed)}.
    Returns {spearman, position_mae, n}.
    """
    common = sorted(set(predicted) & set(actual))
    n = len(common)
    if n == 0:
        return {"spearman": None, "position_mae": None, "n": 0}
    pred = [predicted[c] for c in common]
    act = [actual[c] for c in common]
    mae = _mean([abs(p - a) for p, a in zip(pred, act)])
    return {"spearman": spearman(pred, act), "position_mae": mae, "n": n}
