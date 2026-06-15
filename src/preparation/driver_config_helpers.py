"""
Pure helpers for the driver-config generator
============================================
No pandas / no I/O — unit-testable in isolation on any machine.
"""
from __future__ import annotations

# ConstructorId -> canonical team key. Historical lineages are folded onto the
# current key so the simulator's team grouping is consistent across eras.
CONSTRUCTOR_TO_TEAM = {
    "alfa": "sauber",          # Alfa Romeo -> Kick Sauber lineage
    "alphatauri": "rb",        # AlphaTauri -> Racing Bulls lineage
}


def parse_laptime(value) -> float | None:
    """Parse 'M:SS.mmm' or plain seconds into seconds. Blank/None -> None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if ":" in s:
        mins, secs = s.split(":", 1)
        try:
            return int(mins) * 60 + float(secs)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def best_quali_time(q1, q2, q3) -> float | None:
    """Best quali time by session priority Q3 > Q2 > Q1 (first one set)."""
    for q in (q3, q2, q1):
        t = parse_laptime(q)
        if t is not None:
            return t
    return None


def minmax_normalize(values, lo: float, hi: float) -> list:
    """Min-max scale values into [lo, hi]. All-equal -> midpoint."""
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        mid = round((lo + hi) / 2, 3)
        return [mid for _ in values]
    span = vmax - vmin
    return [lo + (v - vmin) * (hi - lo) / span for v in values]


def constructor_to_team(constructor_id: str) -> str:
    """Map a Jolpica constructorId to a canonical team key."""
    return CONSTRUCTOR_TO_TEAM.get(constructor_id, constructor_id)


def driver_name_from_id(driver_id: str) -> str:
    """Best-effort display name from a Jolpica driverId."""
    return driver_id.replace("_", " ").title()
