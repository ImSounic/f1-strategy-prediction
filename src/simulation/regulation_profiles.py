"""
Regulation-era profiles
=======================
F1 has periodic regulation resets that change the physics the simulator assumes
(tyre construction, fuel/energy, aerodynamics/overtaking). Reusing one era's
constants across a reset biases predictions. A ``RegulationProfile`` bundles the
era-specific constants so the simulator is parameterised by season instead of
hardcoding one era.

Eras:
  ground_effect_2022_25 — ground-effect cars, DRS, C1-C6 tyres (2022-2025).
  new_era_2026          — 2026 reset: active aero + override boost (no DRS),
                          C1-C5 tyres, ~50/50 hybrid PU. The physics constants
                          here are SEEDED from the 2022-25 baseline and are
                          CALIBRATED against real 2026 data in Phase 4.

The 2022-25 profile reproduces the multi-car simulator's original constants
exactly, so existing behaviour (and validation on 2022-25) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegulationProfile:
    """Era-specific physics constants for the multi-car simulator."""
    name: str
    seasons: tuple              # seasons this era covers
    compound_set: tuple         # dry compound codes available (C1..)
    base_pace: float            # reference lap time (s)
    start_fuel_kg: float
    fuel_effect_per_kg: float   # s/lap per kg of fuel on board
    sc_pace_factor: float       # lap-time multiplier under full SC
    vsc_pace_factor: float      # lap-time multiplier under VSC
    compound_deg_base: dict     # {compound_name: base deg rate s/lap}
    compound_cliff: dict        # {compound_name: cliff lap}
    compound_pace_offset: dict  # {compound_name: s/lap vs MEDIUM (SOFT faster fresh)}
    compound_deg_multiplier: dict  # {compound_name: x base deg (SOFT degrades more)}
    dirty_air_window: float     # gap (s) within which dirty air bites
    dirty_air_penalty: float    # max s/lap lost in dirty air
    drs_window: float           # gap (s) within which DRS/override applies
    overtake_aid_benefit: float # s/lap benefit from DRS/override at full effect
    lap_time_noise_std: float   # per-lap gaussian noise std (s)
    overtaking_mode: str        # "drs" | "override_boost"


GROUND_EFFECT_2022_25 = RegulationProfile(
    name="ground_effect_2022_25",
    seasons=(2022, 2023, 2024, 2025),
    compound_set=("C1", "C2", "C3", "C4", "C5", "C6"),
    base_pace=90.0,
    start_fuel_kg=110.0,
    fuel_effect_per_kg=0.035,
    sc_pace_factor=1.40,
    vsc_pace_factor=1.20,
    compound_deg_base={"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04},
    compound_cliff={"SOFT": 20, "MEDIUM": 30, "HARD": 40},
    # Pace offset left neutral: stint lap-time proxies are too confounded by
    # fuel/usage to support a per-compound pace delta (a future clean-air,
    # fuel-corrected calibration could populate it; Phase 4 may do so for 2026).
    compound_pace_offset={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    # Deg multiplier calibrated from data — median DegSlope by compound across
    # 2022-25: SOFT 0.089 / MEDIUM 0.065 / HARD 0.058 -> ratios vs MEDIUM.
    compound_deg_multiplier={"SOFT": 1.38, "MEDIUM": 1.0, "HARD": 0.89},
    dirty_air_window=1.5,
    dirty_air_penalty=0.15,
    drs_window=1.0,
    overtake_aid_benefit=0.3,
    lap_time_noise_std=0.3,
    overtaking_mode="drs",
)

# 2026 reset. Structural changes are known (C6 dropped; override boost replaces
# DRS; closer following from -55% drag). Numeric physics constants are seeded
# from 2022-25 and CALIBRATED against real 2026 data in Phase 4 — only the
# dirty-air penalty is pre-reduced as a placeholder to reflect closer following.
NEW_ERA_2026 = RegulationProfile(
    name="new_era_2026",
    seasons=(2026,),
    compound_set=("C1", "C2", "C3", "C4", "C5"),
    base_pace=90.0,
    start_fuel_kg=110.0,
    fuel_effect_per_kg=0.035,
    sc_pace_factor=1.40,
    vsc_pace_factor=1.20,
    compound_deg_base={"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04},
    compound_cliff={"SOFT": 20, "MEDIUM": 30, "HARD": 40},
    # Pace offset left neutral: stint lap-time proxies are too confounded by
    # fuel/usage to support a per-compound pace delta (a future clean-air,
    # fuel-corrected calibration could populate it; Phase 4 may do so for 2026).
    compound_pace_offset={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    # Deg multiplier calibrated from data — median DegSlope by compound across
    # 2022-25: SOFT 0.089 / MEDIUM 0.065 / HARD 0.058 -> ratios vs MEDIUM.
    compound_deg_multiplier={"SOFT": 1.38, "MEDIUM": 1.0, "HARD": 0.89},
    dirty_air_window=1.0,
    dirty_air_penalty=0.07,
    drs_window=1.0,
    overtake_aid_benefit=0.3,
    lap_time_noise_std=0.3,
    overtaking_mode="override_boost",
)

ERA_PROFILES = (GROUND_EFFECT_2022_25, NEW_ERA_2026)
DEFAULT_PROFILE = GROUND_EFFECT_2022_25


def get_era(season: int) -> str:
    """Return the regulation-era name for a season."""
    if season >= 2026:
        return "new_era_2026"
    return "ground_effect_2022_25"


def get_profile(season: int) -> RegulationProfile:
    """Return the RegulationProfile for a season."""
    name = get_era(season)
    for profile in ERA_PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"No regulation profile for season {season}")
