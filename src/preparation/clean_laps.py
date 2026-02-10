"""
Lap Data Cleaning & Fuel Correction
=====================================
Pipeline:
    1. Load raw FastF1 laps across all seasons
    2. Filter: remove pit in/out laps, inaccurate laps, VSC/SC laps, lap 1
    3. Fuel correction: subtract estimated fuel-mass time benefit per lap
    4. Merge stint info (compound, tyre age) from OpenF1 + FastF1
    5. Output: single consolidated parquet with clean, fuel-corrected lap times

Key columns added:
    - FuelCorrectedLapTime: LapTime minus fuel effect
    - FuelLoad_kg: estimated fuel remaining at each lap
    - IsClean: True if lap passes all quality filters
    - StintNumber: sequential stint number per driver per race

Output:
    data/processed/clean_laps.parquet

Usage:
    python -m src.preparation.clean_laps
    python -m src.preparation.clean_laps --seasons 2024
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Step 1: Load Raw Laps ─────────────────────────────────────────────────

def load_all_laps(laps_dir: Path, seasons: list = None) -> pd.DataFrame:
    """Load and concatenate all raw lap parquet files."""
    frames = []
    for f in sorted(laps_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if seasons and df["Season"].iloc[0] not in seasons:
            continue
        frames.append(df)
    
    if not frames:
        raise ValueError(f"No lap files found in {laps_dir}")
    
    laps = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(laps):,} raw laps from {len(frames)} races")
    return laps


# ── Step 2: Load Track Status ─────────────────────────────────────────────

def load_all_track_status(status_dir: Path, seasons: list = None) -> pd.DataFrame:
    """Load track status data for SC/VSC detection."""
    frames = []
    for f in sorted(status_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if seasons and df["Season"].iloc[0] not in seasons:
            continue
        frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


# ── Step 3: Load Circuit Info ─────────────────────────────────────────────

def load_circuit_info(csv_path: Path) -> pd.DataFrame:
    """Load Pirelli circuit characteristics."""
    return pd.read_csv(csv_path)


# ── Step 4: Quality Filters ──────────────────────────────────────────────

def apply_quality_filters(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters and mark laps as clean/dirty.
    
    Filters:
        1. Must have a valid LapTime (not NaN)
        2. IsAccurate == True (FastF1's own accuracy flag)
        3. Not lap 1 (standing start, not representative)
        4. Not a pit in-lap (PitInTime is set)
        5. Not a pit out-lap (PitOutTime is set)
        6. Not under Safety Car or VSC (TrackStatus contains 4, 5, 6)
        7. Not deleted by stewards
        8. LapTime within reasonable bounds (> 60s, < 200s)
    """
    n_total = len(laps)
    
    # Initialize all as clean
    laps["IsClean"] = True
    
    # Filter 1: Valid LapTime
    mask_no_time = laps["LapTime"].isna()
    laps.loc[mask_no_time, "IsClean"] = False
    
    # Filter 2: IsAccurate flag
    if "IsAccurate" in laps.columns:
        mask_inaccurate = laps["IsAccurate"] == False
        laps.loc[mask_inaccurate, "IsClean"] = False
    
    # Filter 3: Lap 1 (standing start)
    mask_lap1 = laps["LapNumber"] == 1
    laps.loc[mask_lap1, "IsClean"] = False
    
    # Filter 4: Pit in-laps
    if "PitInTime" in laps.columns:
        mask_pit_in = laps["PitInTime"].notna()
    elif "PitInTime" in laps.columns:
        mask_pit_in = laps["PitInTime"].notna()
    else:
        mask_pit_in = pd.Series(False, index=laps.index)
    laps.loc[mask_pit_in, "IsClean"] = False
    
    # Filter 5: Pit out-laps
    if "PitOutTime" in laps.columns:
        mask_pit_out = laps["PitOutTime"].notna()
    elif "PitOutTime" in laps.columns:
        mask_pit_out = laps["PitOutTime"].notna()
    else:
        mask_pit_out = pd.Series(False, index=laps.index)
    laps.loc[mask_pit_out, "IsClean"] = False
    
    # Filter 6: Safety Car / VSC / Red Flag laps
    if "TrackStatus" in laps.columns:
        sc_statuses = {"4", "5", "6", "7"}  # SC, Red, VSC, VSC Ending
        mask_sc = laps["TrackStatus"].astype(str).apply(
            lambda x: any(c in sc_statuses for c in str(x))
        )
        laps.loc[mask_sc, "IsClean"] = False
    
    # Filter 7: Deleted laps
    if "Deleted" in laps.columns:
        mask_deleted = laps["Deleted"] == True
        laps.loc[mask_deleted, "IsClean"] = False
    
    # Filter 8: Reasonable time bounds
    mask_bounds = (laps["LapTime"] < 60) | (laps["LapTime"] > 200)
    laps.loc[mask_bounds, "IsClean"] = False
    
    n_clean = laps["IsClean"].sum()
    logger.info(
        f"Quality filters: {n_clean:,}/{n_total:,} clean laps "
        f"({100*n_clean/n_total:.1f}%)"
    )
    
    return laps


# ── Step 5: Fuel Correction ──────────────────────────────────────────────

def apply_fuel_correction(
    laps: pd.DataFrame,
    circuit_info: pd.DataFrame,
    fuel_config: dict,
) -> pd.DataFrame:
    """
    Apply fuel correction to lap times.
    
    Model:
        fuel_remaining(lap) = start_fuel - burn_rate * lap_number
        fuel_time_effect(lap) = fuel_remaining(lap) * fuel_effect_per_kg
        corrected_time = raw_time - fuel_time_effect(lap)
    
    This normalizes all lap times to an "empty tank" baseline, isolating
    tyre degradation from fuel burn effects.
    """
    start_fuel = fuel_config["start_fuel_kg"]
    burn_rate = fuel_config["burn_rate_kg_per_lap"]
    effect_per_kg = fuel_config["fuel_effect_per_kg_seconds"]
    
    # Merge total_laps from circuit info to compute per-race burn rate
    circuit_merge = circuit_info[["season", "round_number", "total_laps"]].rename(
        columns={"season": "Season", "round_number": "RoundNumber"}
    )
    
    laps = laps.merge(circuit_merge, on=["Season", "RoundNumber"], how="left")
    
    # Compute fuel load at each lap
    # Use race-specific burn rate: start_fuel / total_laps
    laps["BurnRate_kg_per_lap"] = np.where(
        laps["total_laps"].notna() & (laps["total_laps"] > 0),
        start_fuel / laps["total_laps"],
        burn_rate  # fallback to config default
    )
    
    laps["FuelLoad_kg"] = np.maximum(
        0, start_fuel - laps["BurnRate_kg_per_lap"] * (laps["LapNumber"] - 1)
    )
    
    # Fuel time effect: how much slower the car is due to fuel weight
    laps["FuelEffect_seconds"] = laps["FuelLoad_kg"] * effect_per_kg
    
    # Fuel-corrected lap time (subtract the fuel penalty → faster = lighter)
    laps["FuelCorrectedLapTime"] = laps["LapTime"] - laps["FuelEffect_seconds"]
    
    # Also correct sector times if available
    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if sector in laps.columns:
            laps[f"FuelCorrected_{sector}"] = (
                laps[sector] - laps["FuelEffect_seconds"] / 3  # distribute evenly
            )
    
    # Clean up temp columns
    laps.drop(columns=["total_laps", "BurnRate_kg_per_lap"], inplace=True, errors="ignore")
    
    logger.info(
        f"Fuel correction applied: avg correction = "
        f"{laps['FuelEffect_seconds'].mean():.3f}s, "
        f"range [{laps['FuelEffect_seconds'].min():.3f}, "
        f"{laps['FuelEffect_seconds'].max():.3f}]s"
    )
    
    return laps


# ── Step 6: Stint Numbering ──────────────────────────────────────────────

def compute_stint_numbers(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sequential stint numbers per driver per race.
    
    A new stint begins after a pit out-lap. We use the Stint column from
    FastF1 if available, otherwise derive from pit stop events.
    """
    if "Stint" in laps.columns and laps["Stint"].notna().any():
        laps["StintNumber"] = laps["Stint"].astype("Int64")
        logger.info("Stint numbers: using FastF1 Stint column")
    else:
        # Derive from pit stops
        laps = laps.sort_values(["Season", "RoundNumber", "Driver", "LapNumber"])
        
        pit_out_col = "PitOutTime" if "PitOutTime" in laps.columns else "PitOutTime"
        if pit_out_col in laps.columns:
            laps["_is_pit_out"] = laps[pit_out_col].notna().astype(int)
        else:
            laps["_is_pit_out"] = 0
        
        laps["StintNumber"] = (
            laps.groupby(["Season", "RoundNumber", "Driver"])["_is_pit_out"]
            .cumsum() + 1
        ).astype("Int64")
        
        laps.drop(columns=["_is_pit_out"], inplace=True)
        logger.info("Stint numbers: derived from pit stop events")
    
    return laps


# ── Step 7: Compound Mapping ─────────────────────────────────────────────

def map_compound_numbers(
    laps: pd.DataFrame,
    circuit_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map compound names (SOFT/MEDIUM/HARD) to actual compound numbers (C1-C6)
    using the Pirelli allocation data, and add a numeric compound hardness score.
    """
    compound_map = {}
    for _, row in circuit_info.iterrows():
        key = (int(row["season"]), int(row["round_number"]))
        compound_map[key] = {
            "HARD": row["hard_compound"],
            "MEDIUM": row["medium_compound"],
            "SOFT": row["soft_compound"],
        }
    
    def get_actual_compound(row):
        key = (int(row["Season"]), int(row["RoundNumber"]))
        mapping = compound_map.get(key, {})
        compound = str(row.get("Compound", "")).upper()
        return mapping.get(compound, compound)
    
    laps["ActualCompound"] = laps.apply(get_actual_compound, axis=1)
    
    # Numeric hardness: C1=1 (hardest) ... C6=6 (softest)
    hardness_map = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6}
    laps["CompoundHardness"] = laps["ActualCompound"].map(hardness_map)
    
    mapped = laps["CompoundHardness"].notna().sum()
    logger.info(f"Compound mapping: {mapped:,}/{len(laps):,} laps mapped to C1-C6")
    
    return laps


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_cleaning(seasons: list = None, config_path: str = "configs/config.yaml"):
    """Run the full lap cleaning pipeline."""
    config = load_config(config_path)
    
    laps_dir = Path(config["paths"]["raw"]["fastf1"]) / "laps"
    status_dir = Path(config["paths"]["raw"]["fastf1"]) / "track_status"
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    output_dir = Path(config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fuel_config = config["modeling"]["fuel_model"]
    
    # Load data
    logger.info("=" * 60)
    logger.info("  LAP DATA CLEANING & FUEL CORRECTION")
    logger.info("=" * 60)
    
    laps = load_all_laps(laps_dir, seasons)
    circuit_info = load_circuit_info(circuit_csv)
    
    # Apply pipeline
    laps = apply_quality_filters(laps)
    laps = apply_fuel_correction(laps, circuit_info, fuel_config)
    laps = compute_stint_numbers(laps)
    laps = map_compound_numbers(laps, circuit_info)
    
    # Save
    output_path = output_dir / "clean_laps.parquet"
    laps.to_parquet(output_path, index=False)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    clean = laps[laps["IsClean"]]
    logger.info(f"  Total laps:     {len(laps):,}")
    logger.info(f"  Clean laps:     {len(clean):,} ({100*len(clean)/len(laps):.1f}%)")
    logger.info(f"  Seasons:        {sorted(laps['Season'].unique())}")
    logger.info(f"  Races:          {laps.groupby(['Season','RoundNumber']).ngroups}")
    logger.info(f"  Compounds:      {sorted(clean['ActualCompound'].dropna().unique())}")
    logger.info(f"  Avg fuel corr:  {clean['FuelEffect_seconds'].mean():.3f}s")
    logger.info(f"  Output:         {output_path}")
    logger.info(f"  File size:      {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Clean lap data and apply fuel correction.")
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_cleaning(seasons=args.seasons, config_path=args.config)


if __name__ == "__main__":
    main()
