"""
OpenF1 API Client
==================
Extracts supplementary data from OpenF1 (2023 onwards):
    - Stints (compound, tyre_age_at_start, lap_start, lap_end)
    - Intervals (gap to leader, gap to car ahead)
    - Pit stops (precise durations)

Output:
    data/raw/openf1/
    ├── stints.parquet
    ├── intervals.parquet
    └── pitstops.parquet

Usage:
    python -m src.ingestion.openf1_client
    python -m src.ingestion.openf1_client --seasons 2024 2025
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
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


def api_get(base_url: str, endpoint: str, params: dict = None) -> list:
    """Make a GET request to OpenF1 API with retry on rate limit."""
    url = f"{base_url}/{endpoint}"
    
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"  ⏳ OpenF1 rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(0.3)
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning(f"  ⏳ Timeout on {endpoint}, retrying...")
            time.sleep(5)
    
    logger.error(f"  ✗ Failed after 5 retries: {endpoint}")
    return []


def get_sessions(base_url: str, year: int) -> list:
    """Get all race sessions for a given year."""
    sessions = api_get(base_url, "sessions", {
        "year": year,
        "session_type": "Race",
    })
    return sessions or []


def extract_stints(base_url: str, sessions: list) -> pd.DataFrame:
    """Extract stint data for all sessions."""
    logger.info("  Extracting stints...")
    all_rows = []
    
    for sess in sessions:
        session_key = sess.get("session_key")
        year = sess.get("year", "")
        meeting_name = sess.get("meeting_name", "")
        
        data = api_get(base_url, "stints", {"session_key": session_key})
        
        for row in data:
            row["year"] = year
            row["meeting_name"] = meeting_name
            all_rows.append(row)
        
        logger.debug(f"    {year} {meeting_name}: {len(data)} stints")
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    logger.info(f"    Total: {len(df):,} stint records")
    return df


def extract_intervals(base_url: str, sessions: list) -> pd.DataFrame:
    """Extract interval/gap data for all sessions.
    
    Note: Intervals data can be very large (updates every few seconds).
    We sample to keep file sizes reasonable.
    """
    logger.info("  Extracting intervals...")
    all_rows = []
    
    for sess in sessions:
        session_key = sess.get("session_key")
        year = sess.get("year", "")
        meeting_name = sess.get("meeting_name", "")
        
        data = api_get(base_url, "intervals", {"session_key": session_key})
        
        for row in data:
            row["year"] = year
            row["meeting_name"] = meeting_name
            all_rows.append(row)
        
        logger.debug(f"    {year} {meeting_name}: {len(data)} interval records")
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    logger.info(f"    Total: {len(df):,} interval records")
    return df


def extract_pit(base_url: str, sessions: list) -> pd.DataFrame:
    """Extract pit stop data for all sessions."""
    logger.info("  Extracting pit stops...")
    all_rows = []
    
    for sess in sessions:
        session_key = sess.get("session_key")
        year = sess.get("year", "")
        meeting_name = sess.get("meeting_name", "")
        
        data = api_get(base_url, "pit", {"session_key": session_key})
        
        for row in data:
            row["year"] = year
            row["meeting_name"] = meeting_name
            all_rows.append(row)
        
        logger.debug(f"    {year} {meeting_name}: {len(data)} pit stops")
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    logger.info(f"    Total: {len(df):,} pit stop records")
    return df


def run_extraction(seasons: list = None, config_path: str = "configs/config.yaml"):
    """Run the full OpenF1 extraction pipeline."""
    config = load_config(config_path)
    openf1_cfg = config["extraction"]["openf1"]
    
    base_url = openf1_cfg["base_url"]
    if seasons is None:
        seasons = openf1_cfg["seasons"]
    
    output_dir = Path(config["paths"]["raw"]["openf1"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"OpenF1 extraction: seasons={seasons}")
    logger.info(f"Output directory: {output_dir}")
    start = time.time()
    
    # Get all race sessions across requested seasons
    all_sessions = []
    for year in seasons:
        sessions = get_sessions(base_url, year)
        logger.info(f"  {year}: {len(sessions)} race sessions")
        all_sessions.extend(sessions)
    
    if not all_sessions:
        logger.warning("  No sessions found. Exiting.")
        return
    
    # Extract each endpoint
    endpoints = openf1_cfg.get("endpoints", ["stints", "intervals", "pit"])
    
    if "stints" in endpoints:
        stints_df = extract_stints(base_url, all_sessions)
        if not stints_df.empty:
            stints_df.to_parquet(output_dir / "stints.parquet", index=False)
            logger.info(f"  ✓ stints.parquet ({len(stints_df):,} rows)")
    
    if "pit" in endpoints:
        pit_df = extract_pit(base_url, all_sessions)
        if not pit_df.empty:
            pit_df.to_parquet(output_dir / "pitstops.parquet", index=False)
            logger.info(f"  ✓ pitstops.parquet ({len(pit_df):,} rows)")
    
    if "intervals" in endpoints:
        logger.info("  ⚠ Intervals data is large — extracting (this may take a while)...")
        intervals_df = extract_intervals(base_url, all_sessions)
        if not intervals_df.empty:
            intervals_df.to_parquet(output_dir / "intervals.parquet", index=False)
            logger.info(f"  ✓ intervals.parquet ({len(intervals_df):,} rows)")
    
    elapsed = time.time() - start
    logger.info(f"\n  OpenF1 extraction complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Extract F1 data from OpenF1 API.")
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_extraction(seasons=args.seasons, config_path=args.config)


if __name__ == "__main__":
    main()
