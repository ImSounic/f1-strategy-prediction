"""
Feature Engineering Pipeline
==============================
Builds on clean_laps.parquet to create modeling-ready features:

1. Tyre Degradation Features (Time Series):
    - Savitzky-Golay filtered degradation curves per stint
    - Degradation rate (slope) per stint
    - Degradation acceleration (curvature)
    - Stint-level aggregates (mean pace, variance, cliff detection)

2. Race Context Features:
    - Safety car probability features (from track status history)
    - Weather features (track temp, air temp, rainfall rolling stats)
    - Track position / dirty air proxy

3. Driver & Team Performance Features:
    - Rolling form (last N races)
    - Qualifying pace delta to team-mate
    - Constructor championship position

Output:
    data/features/stint_features.parquet   (one row per stint)
    data/features/lap_features.parquet     (one row per lap, enriched)

Usage:
    python -m src.preparation.feature_engineering
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── 1. Tyre Degradation Features ─────────────────────────────────────────

def extract_degradation_features(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract tyre degradation features per stint using Time Series techniques.
    
    For each stint (driver × race × stint_number):
        1. Take the fuel-corrected lap times (clean laps only)
        2. Apply Savitzky-Golay filter to smooth noise
        3. Compute degradation rate (1st derivative = slope)
        4. Compute degradation acceleration (2nd derivative)
        5. Detect "cliff" (sudden degradation spike)
    
    Returns:
        stint_features: DataFrame with one row per stint
        laps: DataFrame with smoothed columns added per lap
    """
    logger.info("Extracting tyre degradation features...")
    
    # Work with clean laps that have valid compounds
    valid_compounds = {"C1", "C2", "C3", "C4", "C5", "C6"}
    mask = (
        (laps["IsClean"] == True) &
        (laps["ActualCompound"].isin(valid_compounds)) &
        (laps["FuelCorrectedLapTime"].notna())
    )
    clean = laps[mask].copy()
    
    group_cols = ["Season", "RoundNumber", "EventName", "Driver", "Team", "StintNumber"]
    
    stint_records = []
    smoothed_laps = []
    
    for group_key, stint_df in clean.groupby(group_cols, dropna=False):
        stint_df = stint_df.sort_values("LapNumber").copy()
        n_laps = len(stint_df)
        
        season, rnd, event, driver, team, stint_num = group_key
        compound = stint_df["ActualCompound"].iloc[0]
        hardness = stint_df["CompoundHardness"].iloc[0]
        tyre_life_start = stint_df["TyreLife"].iloc[0] if "TyreLife" in stint_df.columns else 0
        
        times = stint_df["FuelCorrectedLapTime"].values
        lap_nums = stint_df["LapNumber"].values
        
        # ── Savitzky-Golay Smoothing ──
        # Window must be odd and <= n_laps; polyorder < window
        if n_laps >= 5:
            window = min(7, n_laps if n_laps % 2 == 1 else n_laps - 1)
            polyorder = min(2, window - 1)
            smoothed = savgol_filter(times, window_length=window, polyorder=polyorder)
            
            # First derivative: degradation rate (seconds per lap)
            deg_rate = savgol_filter(
                times, window_length=window, polyorder=polyorder, deriv=1
            )
            
            # Second derivative: degradation acceleration
            if n_laps >= 7:
                deg_accel = savgol_filter(
                    times, window_length=window, polyorder=polyorder, deriv=2
                )
            else:
                deg_accel = np.zeros_like(times)
        elif n_laps >= 3:
            # Minimal smoothing for short stints
            smoothed = savgol_filter(times, window_length=3, polyorder=1)
            deg_rate = np.gradient(smoothed)
            deg_accel = np.zeros_like(times)
        else:
            # Too short to smooth
            smoothed = times.copy()
            deg_rate = np.zeros_like(times)
            deg_accel = np.zeros_like(times)
        
        # Add smoothed values back to lap-level data
        stint_df["SmoothedLapTime"] = smoothed
        stint_df["DegradationRate"] = deg_rate
        stint_df["DegradationAccel"] = deg_accel
        smoothed_laps.append(stint_df)
        
        # ── Stint-Level Features ──
        # Linear regression for overall degradation slope
        if n_laps >= 3:
            tyre_ages = np.arange(n_laps)
            slope, intercept = np.polyfit(tyre_ages, times, 1)
            residuals = times - (intercept + slope * tyre_ages)
            
            # Quadratic fit for curvature
            if n_laps >= 5:
                quad_coeffs = np.polyfit(tyre_ages, times, 2)
                curvature = quad_coeffs[0]  # coefficient of x²
            else:
                curvature = 0.0
        else:
            slope = 0.0
            intercept = times[0] if n_laps > 0 else np.nan
            curvature = 0.0
            residuals = np.zeros_like(times)
        
        # Cliff detection: is there a lap where deg_rate spikes > 2x mean?
        mean_deg = np.mean(np.abs(deg_rate)) if n_laps >= 3 else 0
        cliff_detected = bool(np.any(deg_rate > max(2 * mean_deg, 0.15))) if n_laps >= 5 else False
        cliff_lap = int(lap_nums[np.argmax(deg_rate)]) if cliff_detected else None
        
        stint_records.append({
            "Season": season,
            "RoundNumber": rnd,
            "EventName": event,
            "Driver": driver,
            "Team": team,
            "StintNumber": stint_num,
            "Compound": compound,
            "CompoundHardness": hardness,
            "TyreLifeStart": tyre_life_start,
            "StintLength": n_laps,
            "LapStart": int(lap_nums[0]),
            "LapEnd": int(lap_nums[-1]),
            # Pace
            "MeanLapTime": np.mean(times),
            "MedianLapTime": np.median(times),
            "BestLapTime": np.min(times),
            "StdLapTime": np.std(times),
            # Degradation (Time Series features)
            "DegSlope": slope,
            "DegIntercept": intercept,
            "DegCurvature": curvature,
            "DegResidualStd": np.std(residuals),
            "MeanDegRate": np.mean(deg_rate),
            "MaxDegRate": np.max(deg_rate) if n_laps >= 3 else 0,
            "DegRateVariance": np.var(deg_rate),
            # Cliff
            "CliffDetected": cliff_detected,
            "CliffLap": cliff_lap,
            # Smoothed deltas
            "SmoothedFirst": smoothed[0] if n_laps > 0 else np.nan,
            "SmoothedLast": smoothed[-1] if n_laps > 0 else np.nan,
            "SmoothedDelta": smoothed[-1] - smoothed[0] if n_laps > 1 else 0,
        })
    
    stint_features = pd.DataFrame(stint_records)
    enriched_laps = pd.concat(smoothed_laps, ignore_index=True) if smoothed_laps else pd.DataFrame()
    
    logger.info(
        f"  {len(stint_features):,} stints extracted | "
        f"Avg deg slope: {stint_features['DegSlope'].mean():.4f} s/lap | "
        f"Cliffs detected: {stint_features['CliffDetected'].sum()}"
    )
    
    return stint_features, enriched_laps


# ── 2. Weather Features ──────────────────────────────────────────────────

def extract_weather_features(
    laps: pd.DataFrame,
    weather_dir: Path,
    seasons: list = None,
) -> pd.DataFrame:
    """
    Merge weather data with lap data.
    
    For each lap, find the nearest weather observation and add:
        - AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, WindDirection
        - Rainfall (boolean)
        - TrackTemp change rate (rolling delta)
    """
    logger.info("Extracting weather features...")
    
    weather_frames = []
    for f in sorted(weather_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if seasons and df["Season"].iloc[0] not in seasons:
            continue
        weather_frames.append(df)
    
    if not weather_frames:
        logger.warning("  No weather data found")
        return laps
    
    weather = pd.concat(weather_frames, ignore_index=True)
    
    # Aggregate weather per race to race-level features
    weather_agg = weather.groupby(["Season", "RoundNumber"]).agg(
        MeanAirTemp=("AirTemp", "mean"),
        MeanTrackTemp=("TrackTemp", "mean"),
        MaxTrackTemp=("TrackTemp", "max"),
        MinTrackTemp=("TrackTemp", "min"),
        TrackTempRange=("TrackTemp", lambda x: x.max() - x.min()),
        MeanHumidity=("Humidity", "mean"),
        MeanWindSpeed=("WindSpeed", "mean"),
        MeanPressure=("Pressure", "mean"),
        RainfallDetected=("Rainfall", lambda x: (x == True).any()),
        RainfallFraction=("Rainfall", lambda x: (x == True).mean()),
    ).reset_index()
    
    laps = laps.merge(weather_agg, on=["Season", "RoundNumber"], how="left")
    
    filled = laps["MeanTrackTemp"].notna().sum()
    logger.info(f"  Weather features merged: {filled:,}/{len(laps):,} laps")
    
    return laps


# ── 3. Safety Car / Incident Features ────────────────────────────────────

def extract_incident_features(
    circuit_info: pd.DataFrame,
    status_dir: Path,
    seasons: list = None,
) -> pd.DataFrame:
    """
    Compute safety car probability features per circuit.
    
    For each circuit, calculate:
        - Historical SC rate (fraction of races with SC)
        - Historical VSC rate
        - Historical red flag rate
        - Average number of SC periods per race
    """
    logger.info("Extracting incident/safety car features...")
    
    frames = []
    for f in sorted(status_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if seasons and df["Season"].iloc[0] not in seasons:
            continue
        frames.append(df)
    
    if not frames:
        logger.warning("  No track status data found")
        return pd.DataFrame()
    
    status = pd.concat(frames, ignore_index=True)
    
    # Map circuit_name from laps to status via Season+RoundNumber
    race_incidents = []
    for (season, rnd), grp in status.groupby(["Season", "RoundNumber"]):
        statuses = set(grp["Status"].astype(str).values)
        
        race_incidents.append({
            "Season": season,
            "RoundNumber": rnd,
            "HasSC": "4" in statuses,
            "HasVSC": "6" in statuses or "7" in statuses,
            "HasRedFlag": "5" in statuses,
            "NumStatusChanges": len(grp),
            "NumSCPeriods": (grp["Status"].astype(str) == "4").sum(),
        })
    
    incidents = pd.DataFrame(race_incidents)
    
    logger.info(
        f"  SC rate: {incidents['HasSC'].mean():.1%} | "
        f"VSC rate: {incidents['HasVSC'].mean():.1%} | "
        f"Red flag rate: {incidents['HasRedFlag'].mean():.1%}"
    )
    
    return incidents


# ── 4. Driver & Team Performance Features ────────────────────────────────

def extract_performance_features(
    results_path: Path,
    qualifying_path: Path,
    standings_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute driver and team performance features:
        - Driver: rolling avg finish position (last 5 races)
        - Driver: qualifying delta to teammate
        - Team: constructor championship points & position
    """
    logger.info("Extracting driver/team performance features...")
    
    results = pd.read_parquet(results_path)
    qualifying = pd.read_parquet(qualifying_path)
    standings = pd.read_parquet(standings_path)
    
    # ── Driver rolling form ──
    results_sorted = results.sort_values(["season", "round"]).copy()
    results_sorted["FinishPos"] = pd.to_numeric(results_sorted["position"], errors="coerce")
    
    driver_form = (
        results_sorted.groupby("driverId")["FinishPos"]
        .apply(lambda x: x.rolling(5, min_periods=1).mean())
    )
    results_sorted["RollingAvgFinish_5"] = driver_form.values
    
    # Points per race rolling
    driver_points = (
        results_sorted.groupby("driverId")["points"]
        .apply(lambda x: x.rolling(5, min_periods=1).mean())
    )
    results_sorted["RollingAvgPoints_5"] = driver_points.values
    
    driver_features = results_sorted[[
        "season", "round", "driverId", "driverCode", "constructorId",
        "grid", "FinishPos", "points", "status",
        "RollingAvgFinish_5", "RollingAvgPoints_5",
    ]].copy()
    
    # ── Constructor strength ──
    team_features = standings.rename(columns={
        "season": "Season",
        "constructorId": "ConstructorId",
        "position": "ConstructorStandingPos",
        "points": "ConstructorPoints",
        "wins": "ConstructorWins",
    })
    
    logger.info(
        f"  Driver features: {len(driver_features):,} entries | "
        f"Team features: {len(team_features):,} entries"
    )
    
    return driver_features, team_features


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_feature_engineering(seasons: list = None, config_path: str = "configs/config.yaml"):
    """Run the full feature engineering pipeline."""
    config = load_config(config_path)
    
    processed_dir = Path(config["paths"]["processed"])
    features_dir = Path(config["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    raw_paths = config["paths"]["raw"]
    
    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    # Load clean laps
    laps = pd.read_parquet(processed_dir / "clean_laps.parquet")
    logger.info(f"Loaded {len(laps):,} laps from clean_laps.parquet")
    
    # 1. Tyre degradation features
    stint_features, enriched_laps = extract_degradation_features(laps)
    
    # 2. Weather features (add to lap-level data)
    weather_dir = Path(raw_paths["fastf1"]) / "weather"
    enriched_laps = extract_weather_features(enriched_laps, weather_dir, seasons)
    
    # 3. Incident features
    status_dir = Path(raw_paths["fastf1"]) / "track_status"
    incidents = extract_incident_features(
        pd.read_csv(Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"),
        status_dir,
        seasons,
    )
    
    # Merge incident features into stint data
    if not incidents.empty:
        stint_features = stint_features.merge(
            incidents, on=["Season", "RoundNumber"], how="left"
        )
    
    # 4. Driver/team performance features
    jolpica_dir = Path(raw_paths["jolpica"])
    driver_features, team_features = extract_performance_features(
        jolpica_dir / "results.parquet",
        jolpica_dir / "qualifying.parquet",
        jolpica_dir / "constructor_standings.parquet",
    )
    
    # Save all features
    stint_features.to_parquet(features_dir / "stint_features.parquet", index=False)
    logger.info(f"  ✓ stint_features.parquet ({len(stint_features):,} rows)")
    
    enriched_laps.to_parquet(features_dir / "lap_features.parquet", index=False)
    logger.info(f"  ✓ lap_features.parquet ({len(enriched_laps):,} rows)")
    
    driver_features.to_parquet(features_dir / "driver_features.parquet", index=False)
    logger.info(f"  ✓ driver_features.parquet ({len(driver_features):,} rows)")
    
    team_features.to_parquet(features_dir / "team_features.parquet", index=False)
    logger.info(f"  ✓ team_features.parquet ({len(team_features):,} rows)")
    
    if not incidents.empty:
        incidents.to_parquet(features_dir / "incident_features.parquet", index=False)
        logger.info(f"  ✓ incident_features.parquet ({len(incidents):,} rows)")
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Stints:          {len(stint_features):,}")
    logger.info(f"  Clean laps:      {len(enriched_laps):,}")
    logger.info(f"  Avg deg slope:   {stint_features['DegSlope'].mean():.4f} s/lap")
    logger.info(f"  Compounds:       {sorted(stint_features['Compound'].unique())}")
    logger.info(f"  Cliffs found:    {stint_features['CliffDetected'].sum()}")
    logger.info(f"  Output dir:      {features_dir}")


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline.")
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_feature_engineering(seasons=args.seasons, config_path=args.config)


if __name__ == "__main__":
    main()
