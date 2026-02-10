"""
FastF1 Data Extraction Pipeline
================================
Extracts lap-level timing, weather, track status, race control messages,
and (optionally) car telemetry for all races in configured seasons.

Output structure:
    data/raw/fastf1/
    ├── laps/          2024_01_Bahrain_Grand_Prix_R.parquet
    ├── weather/       ...
    ├── track_status/  ...
    ├── race_control/  ...
    └── telemetry/     (optional, large files)

Usage:
    python -m src.ingestion.fastf1_extractor
    python -m src.ingestion.fastf1_extractor --seasons 2024 2025
    python -m src.ingestion.fastf1_extractor --seasons 2024 --telemetry
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import fastf1
from fastf1.req import RateLimitExceededError
import pandas as pd
import yaml


# ── Rate Limit Handling ────────────────────────────────────────────────────
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 120  # seconds to wait when rate-limited

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_filename(season: int, round_num: int, event_name: str, session_type: str) -> str:
    safe_name = event_name.replace(" ", "_").replace("/", "-")
    return f"{season}_{round_num:02d}_{safe_name}_{session_type}.parquet"


def save_parquet(df: pd.DataFrame, output_dir: Path, filename: str) -> Optional[Path]:
    if df is None or df.empty:
        logger.warning(f"  ⚠ Empty DataFrame — skipping {filename}")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    # Convert timedelta columns to float seconds for Parquet compatibility
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
    
    df.to_parquet(filepath, engine="pyarrow", index=False)
    logger.info(f"  ✓ Saved {filename} ({len(df):,} rows)")
    return filepath


# ── Core Extraction Functions ──────────────────────────────────────────────

def extract_laps(session, metadata: dict) -> pd.DataFrame:
    """Extract lap-level timing data with metadata columns."""
    laps = session.laps
    if laps.empty:
        return pd.DataFrame()
    
    keep_cols = [
        "Driver", "DriverNumber", "Team",
        "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "Compound", "TyreLife", "FreshTyre", "Stint",
        "PitInTime", "PitOutTime",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Position", "TrackStatus", "IsAccurate",
        "Deleted", "DeletedReason",
    ]
    
    available = [c for c in keep_cols if c in laps.columns]
    df = laps[available].copy()
    
    df["Season"] = metadata["season"]
    df["RoundNumber"] = metadata["round_num"]
    df["EventName"] = metadata["event_name"]
    df["CircuitKey"] = metadata.get("circuit_key", "")
    df["SessionType"] = metadata["session_type"]
    
    return df


def extract_weather(session, metadata: dict) -> pd.DataFrame:
    """Extract weather time series (~1 sample/min)."""
    weather = session.weather_data
    if weather is None or weather.empty:
        return pd.DataFrame()
    
    df = weather.copy()
    df["Season"] = metadata["season"]
    df["RoundNumber"] = metadata["round_num"]
    df["EventName"] = metadata["event_name"]
    df["SessionType"] = metadata["session_type"]
    
    return df


def extract_track_status(session, metadata: dict) -> pd.DataFrame:
    """Extract track status changes (1=Clear, 2=Yellow, 4=SC, 5=Red, 6=VSC, 7=VSC Ending)."""
    track_status = session.track_status
    if track_status is None or track_status.empty:
        return pd.DataFrame()
    
    df = track_status.copy()
    df["Season"] = metadata["season"]
    df["RoundNumber"] = metadata["round_num"]
    df["EventName"] = metadata["event_name"]
    df["SessionType"] = metadata["session_type"]
    
    return df


def extract_race_control(session, metadata: dict) -> pd.DataFrame:
    """Extract race control messages (flags, DRS, penalties)."""
    try:
        rcm = session.race_control_messages
        if rcm is None or rcm.empty:
            return pd.DataFrame()
    except Exception:
        logger.debug("  Race control messages not available for this session")
        return pd.DataFrame()
    
    df = rcm.copy()
    df["Season"] = metadata["season"]
    df["RoundNumber"] = metadata["round_num"]
    df["EventName"] = metadata["event_name"]
    df["SessionType"] = metadata["session_type"]
    
    return df


def extract_telemetry(session, metadata: dict) -> pd.DataFrame:
    """Extract high-frequency car telemetry (3-4 Hz per car). WARNING: Large files."""
    all_telemetry = []
    
    for driver in session.drivers:
        try:
            driver_laps = session.laps.pick_drivers(driver)
            if driver_laps.empty:
                continue
            
            tel = driver_laps.get_car_data()
            if tel is None or tel.empty:
                continue
            
            tel = tel.copy()
            tel["Driver"] = driver
            try:
                tel["DriverAbbr"] = driver_laps.iloc[0]["Driver"]
            except (KeyError, IndexError):
                tel["DriverAbbr"] = driver
            
            all_telemetry.append(tel)
        except Exception as e:
            logger.debug(f"  Telemetry unavailable for driver {driver}: {e}")
    
    if not all_telemetry:
        return pd.DataFrame()
    
    df = pd.concat(all_telemetry, ignore_index=True)
    df["Season"] = metadata["season"]
    df["RoundNumber"] = metadata["round_num"]
    df["EventName"] = metadata["event_name"]
    df["SessionType"] = metadata["session_type"]
    
    return df


# ── Main Extraction Pipeline ──────────────────────────────────────────────

def load_session_with_retry(season, round_num, session_type, include_telemetry):
    """Load a FastF1 session with automatic retry on rate limit."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(season, round_num, session_type)
            session.load(
                laps=True,
                weather=True,
                messages=True,
                telemetry=include_telemetry,
            )
            return session
        except RateLimitExceededError:
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"  ⏳ Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {RATE_LIMIT_WAIT}s..."
                )
                time.sleep(RATE_LIMIT_WAIT)
            else:
                logger.error(f"  ✗ Rate limit exceeded after {MAX_RETRIES} retries. Skipping.")
                return None
        except Exception as e:
            logger.error(f"  ✗ Failed to load session: {e}")
            return None
    return None


def extract_session(
    season: int,
    round_num: int,
    event_name: str,
    session_type: str,
    base_dir: Path,
    include_telemetry: bool = False,
) -> dict:
    """Extract all data for a single session with rate limit handling."""
    results = {}
    filename = make_filename(season, round_num, event_name, session_type)
    
    metadata = {
        "season": season,
        "round_num": round_num,
        "event_name": event_name,
        "session_type": session_type,
    }
    
    logger.info(f"Loading {season} R{round_num:02d} {event_name} ({session_type})...")
    session = load_session_with_retry(season, round_num, session_type, include_telemetry)
    
    if session is None:
        return results
    
    try:
        metadata["circuit_key"] = str(session.event.get("CircuitKey", ""))
    except Exception:
        metadata["circuit_key"] = ""
    
    # Extract each data type — skip gracefully if data didn't load
    try:
        laps_df = extract_laps(session, metadata)
        results["laps"] = save_parquet(laps_df, base_dir / "laps", filename)
    except Exception as e:
        logger.warning(f"  ⚠ Laps extraction failed: {e}")
    
    try:
        weather_df = extract_weather(session, metadata)
        results["weather"] = save_parquet(weather_df, base_dir / "weather", filename)
    except Exception as e:
        logger.warning(f"  ⚠ Weather extraction failed: {e}")
    
    try:
        status_df = extract_track_status(session, metadata)
        results["track_status"] = save_parquet(status_df, base_dir / "track_status", filename)
    except Exception as e:
        logger.warning(f"  ⚠ Track status extraction failed: {e}")
    
    try:
        rcm_df = extract_race_control(session, metadata)
        results["race_control"] = save_parquet(rcm_df, base_dir / "race_control", filename)
    except Exception as e:
        logger.warning(f"  ⚠ Race control extraction failed: {e}")
    
    if include_telemetry:
        try:
            tel_df = extract_telemetry(session, metadata)
            results["telemetry"] = save_parquet(tel_df, base_dir / "telemetry", filename)
        except Exception as e:
            logger.warning(f"  ⚠ Telemetry extraction failed: {e}")
    
    return results


def extract_season(
    season: int,
    base_dir: Path,
    session_types: list,
    include_telemetry: bool = False,
) -> list:
    """Extract all races in a season."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  SEASON {season}")
    logger.info(f"{'='*60}")
    
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    race_events = schedule[schedule["RoundNumber"] > 0]
    
    logger.info(f"Found {len(race_events)} events in {season}")
    
    season_results = []
    
    for _, event in race_events.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = event["EventName"]
        
        for session_type in session_types:
            result = extract_session(
                season=season,
                round_num=round_num,
                event_name=event_name,
                session_type=session_type,
                base_dir=base_dir,
                include_telemetry=include_telemetry,
            )
            season_results.append({
                "season": season,
                "round": round_num,
                "event": event_name,
                "session": session_type,
                "files": result,
            })
            
            time.sleep(2)  # Pause between sessions to respect rate limits
    
    return season_results


def run_extraction(
    seasons: list = None,
    session_types: list = None,
    include_telemetry: bool = False,
    config_path: str = "configs/config.yaml",
):
    """Run the full FastF1 extraction pipeline."""
    config = load_config(config_path)
    extraction_cfg = config["extraction"]["fastf1"]
    
    if seasons is None:
        seasons = config["extraction"]["seasons"]
    if session_types is None:
        session_types = extraction_cfg["session_types"]
    
    base_dir = Path(config["paths"]["raw"]["fastf1"])
    cache_dir = Path(extraction_cfg["cache_dir"])
    
    # Enable FastF1 caching (avoids re-downloads)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info(f"FastF1 cache enabled at: {cache_dir}")
    
    all_results = []
    total_files = 0
    failed_sessions = []
    
    logger.info(f"Starting extraction: seasons={seasons}, sessions={session_types}")
    logger.info(f"Telemetry: {'ENABLED' if include_telemetry else 'DISABLED'}")
    logger.info(f"Output directory: {base_dir}")
    
    start_time = time.time()
    
    for season in seasons:
        season_results = extract_season(
            season=season,
            base_dir=base_dir,
            session_types=session_types,
            include_telemetry=include_telemetry,
        )
        all_results.extend(season_results)
        
        for r in season_results:
            file_count = sum(1 for v in r["files"].values() if v is not None)
            total_files += file_count
            if file_count == 0:
                failed_sessions.append(f"{r['season']} R{r['round']:02d} {r['event']}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Seasons processed:  {len(seasons)}")
    logger.info(f"  Sessions processed: {len(all_results)}")
    logger.info(f"  Files created:      {total_files}")
    logger.info(f"  Time elapsed:       {elapsed/60:.1f} minutes")
    
    if failed_sessions:
        logger.warning(f"  Failed sessions ({len(failed_sessions)}):")
        for fs in failed_sessions:
            logger.warning(f"    - {fs}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Extract F1 data from FastF1 API into Parquet files."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=None,
        help="Seasons to extract (default: from config.yaml)",
    )
    parser.add_argument(
        "--sessions", nargs="+", type=str, default=None,
        help="Session types: R, Q, FP1, FP2, FP3 (default: from config)",
    )
    parser.add_argument(
        "--telemetry", action="store_true",
        help="Include high-frequency car telemetry (WARNING: large files)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    
    args = parser.parse_args()
    
    run_extraction(
        seasons=args.seasons,
        session_types=args.sessions,
        include_telemetry=args.telemetry,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
