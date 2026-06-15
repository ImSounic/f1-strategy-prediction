"""Tests for regulation-era profiles. Pure Python; run: python tests/test_regulation_profiles.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.regulation_profiles import (
    get_era, get_profile, GROUND_EFFECT_2022_25, NEW_ERA_2026, DEFAULT_PROFILE,
)


def test_get_era_boundaries():
    assert get_era(2022) == "ground_effect_2022_25"
    assert get_era(2025) == "ground_effect_2022_25"
    assert get_era(2026) == "new_era_2026"
    assert get_era(2027) == "new_era_2026"


def test_get_profile_returns_matching_era():
    assert get_profile(2024) is GROUND_EFFECT_2022_25
    assert get_profile(2026) is NEW_ERA_2026


def test_default_profile_is_2022_25():
    assert DEFAULT_PROFILE is GROUND_EFFECT_2022_25


def test_2022_25_profile_pins_legacy_constants():
    p = GROUND_EFFECT_2022_25
    assert p.base_pace == 90.0
    assert p.start_fuel_kg == 110.0
    assert p.fuel_effect_per_kg == 0.035
    assert p.sc_pace_factor == 1.40
    assert p.vsc_pace_factor == 1.20
    assert p.compound_deg_base == {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04}
    assert p.compound_cliff == {"SOFT": 20, "MEDIUM": 30, "HARD": 40}
    assert p.dirty_air_window == 1.5
    assert p.dirty_air_penalty == 0.15
    assert p.drs_window == 1.0
    assert p.overtake_aid_benefit == 0.3
    assert p.lap_time_noise_std == 0.3
    assert p.compound_set == ("C1", "C2", "C3", "C4", "C5", "C6")
    assert p.overtaking_mode == "drs"


def test_2026_profile_drops_c6_and_changes_overtaking():
    p = NEW_ERA_2026
    assert "C6" not in p.compound_set
    assert p.compound_set == ("C1", "C2", "C3", "C4", "C5")
    assert p.overtaking_mode == "override_boost"
    assert p.dirty_air_penalty < GROUND_EFFECT_2022_25.dirty_air_penalty


def test_profiles_have_compound_pace_and_deg_fields():
    for p in (GROUND_EFFECT_2022_25, NEW_ERA_2026):
        assert set(p.compound_pace_offset) == {"SOFT", "MEDIUM", "HARD"}
        assert set(p.compound_deg_multiplier) == {"SOFT", "MEDIUM", "HARD"}
        assert p.compound_pace_offset["SOFT"] < p.compound_pace_offset["HARD"]
        assert p.compound_deg_multiplier["SOFT"] > p.compound_deg_multiplier["HARD"]
        assert p.compound_pace_offset["MEDIUM"] == 0.0
        assert p.compound_deg_multiplier["MEDIUM"] == 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
