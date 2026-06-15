"""
Position-aware strategy selection (pure)
========================================
Helpers for choosing among candidate strategies by a sim-derived objective.
Pure (only depends on the pure strategy_match.normalize_sequence) so it is
unit-testable without the simulator or data.
"""
from __future__ import annotations

from src.analysis.strategy_match import normalize_sequence


def dedupe_by_sequence(strategies):
    """Keep one strategy per distinct compound sequence, preserving order.

    `strategies` is any iterable of objects with a `.compound_sequence`
    attribute (a list like ["MEDIUM", "HARD"] or a "M -> H" string).
    Empty/unknown sequences are skipped.
    """
    seen = set()
    out = []
    for s in strategies:
        key = normalize_sequence(s.compound_sequence)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def argmin_by(items, key):
    """Return the item minimizing item[key], or None if empty."""
    if not items:
        return None
    return min(items, key=lambda d: d[key])
