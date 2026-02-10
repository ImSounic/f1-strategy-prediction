"""
Jolpica/Ergast API Client
==========================
Extracts historical F1 data from the Jolpica API (Ergast successor):
    - Race results (grid, finish position, status, fastest lap)
    - Qualifying results (Q1/Q2/Q3 times)
    - Constructor standings (season-long team performance)
    - Pit stops (lap number, duration per stop)

Output:
    data/raw/jolpica/
    ├── results.parquet
    ├── qualifying.parquet
    ├── constructor_standings.parquet
    └── pitstops.parquet

Usage:
    python -m src.ingestion.jolpica_client
    python -m src.ingestion.jolpica_client --seasons 2024 2025
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

RATE_LIMIT_DELAY = 0.3  # seconds between API calls


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def api_get(base_url: str, path: str, limit: int = 100, offset: int = 0) -> dict:
    """Make a paginated GET request to the Jolpica API with retry on 429."""
    url = f"{base_url}/{path}.json"
    params = {"limit": limit, "offset": offset}
    
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning(f"  ⏳ Jolpica rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
        return resp.json()["MRData"]
    
    raise Exception(f"Jolpica API failed after 5 retries: {path}")


def fetch_all_pages(base_url: str, path: str, limit: int = 100) -> list:
    """Fetch all pages of a paginated Jolpica endpoint."""
    all_data = []
    offset = 0
    
    while True:
        data = api_get(base_url, path, limit=limit, offset=offset)
        total = int(data["total"])
        
        # The actual data table is nested — find it
        table_key = [k for k in data.keys() if k not in ("xmlns", "series", "url", "limit", "offset", "total")]
        if not table_key:
            break
        
        table = data[table_key[0]]
        # The table itself contains a "Races" or similar list
        inner_key = [k for k in table.keys() if k not in ("season", "round", "url")]
        if not inner_key:
            break
        
        rows = table[inner_key[0]]
        all_data.extend(rows)
        
        offset += limit
        if offset >= total:
            break
    
    return all_data


# ── Extraction Functions ──────────────────────────────────────────────────

def extract_results(base_url: str, seasons: list) -> pd.DataFrame:
    """Extract race results for all seasons."""
    logger.info("  Extracting race results...")
    all_rows = []
    
    for season in seasons:
        races = fetch_all_pages(base_url, f"{season}/results", limit=100)
        
        for race in races:
            race_info = {
                "season": int(race.get("season", season)),
                "round": int(race.get("round", 0)),
                "raceName": race.get("raceName", ""),
                "circuitId": race.get("Circuit", {}).get("circuitId", ""),
                "circuitName": race.get("Circuit", {}).get("circuitName", ""),
                "date": race.get("date", ""),
            }
            
            for result in race.get("Results", []):
                row = {
                    **race_info,
                    "driverId": result.get("Driver", {}).get("driverId", ""),
                    "driverCode": result.get("Driver", {}).get("code", ""),
                    "constructorId": result.get("Constructor", {}).get("constructorId", ""),
                    "grid": int(result.get("grid", 0)),
                    "position": int(result.get("position", 0)) if result.get("position", "").isdigit() else None,
                    "positionText": result.get("positionText", ""),
                    "points": float(result.get("points", 0)),
                    "laps": int(result.get("laps", 0)),
                    "status": result.get("status", ""),
                    "fastestLapRank": result.get("FastestLap", {}).get("rank", None),
                    "fastestLapTime": result.get("FastestLap", {}).get("Time", {}).get("time", None),
                    "fastestLapAvgSpeed": result.get("FastestLap", {}).get("AverageSpeed", {}).get("speed", None),
                }
                all_rows.append(row)
        
        logger.info(f"    {season}: {len(races)} races")
    
    return pd.DataFrame(all_rows)


def extract_qualifying(base_url: str, seasons: list) -> pd.DataFrame:
    """Extract qualifying results for all seasons."""
    logger.info("  Extracting qualifying results...")
    all_rows = []
    
    for season in seasons:
        races = fetch_all_pages(base_url, f"{season}/qualifying", limit=100)
        
        for race in races:
            race_info = {
                "season": int(race.get("season", season)),
                "round": int(race.get("round", 0)),
                "raceName": race.get("raceName", ""),
                "circuitId": race.get("Circuit", {}).get("circuitId", ""),
            }
            
            for qual in race.get("QualifyingResults", []):
                row = {
                    **race_info,
                    "driverId": qual.get("Driver", {}).get("driverId", ""),
                    "driverCode": qual.get("Driver", {}).get("code", ""),
                    "constructorId": qual.get("Constructor", {}).get("constructorId", ""),
                    "position": int(qual.get("position", 0)),
                    "Q1": qual.get("Q1", None),
                    "Q2": qual.get("Q2", None),
                    "Q3": qual.get("Q3", None),
                }
                all_rows.append(row)
        
        logger.info(f"    {season}: {len(races)} races")
    
    return pd.DataFrame(all_rows)


def extract_constructor_standings(base_url: str, seasons: list) -> pd.DataFrame:
    """Extract end-of-season constructor standings."""
    logger.info("  Extracting constructor standings...")
    all_rows = []
    
    for season in seasons:
        try:
            standings_data = fetch_all_pages(base_url, f"{season}/constructorStandings", limit=100)
            
            for standing in standings_data:
                # standings_data is a list of StandingsLists
                for entry in standing.get("ConstructorStandings", []):
                    row = {
                        "season": season,
                        "position": int(entry.get("position", 0)),
                        "points": float(entry.get("points", 0)),
                        "wins": int(entry.get("wins", 0)),
                        "constructorId": entry.get("Constructor", {}).get("constructorId", ""),
                        "constructorName": entry.get("Constructor", {}).get("name", ""),
                    }
                    all_rows.append(row)
            
            logger.info(f"    {season}: {len([r for r in all_rows if r['season'] == season])} constructors")
        except Exception as e:
            logger.warning(f"    {season}: Failed — {e}")
    
    return pd.DataFrame(all_rows)


def extract_pitstops(base_url: str, seasons: list, results_df: pd.DataFrame = None) -> pd.DataFrame:
    """Extract pit stop data for all races. Uses results_df to avoid re-fetching race list."""
    logger.info("  Extracting pit stops...")
    all_rows = []
    
    for season in seasons:
        # Get round numbers from already-fetched results instead of re-calling API
        if results_df is not None and len(results_df) > 0:
            season_data = results_df[results_df["season"] == season]
            rounds = sorted(season_data[["round", "raceName"]].drop_duplicates().values.tolist())
        else:
            # Fallback: assume up to 30 rounds
            rounds = [[r, f"Round {r}"] for r in range(1, 31)]
        
        for round_num, race_name in rounds:
            try:
                pitstop_data = fetch_all_pages(
                    base_url, f"{season}/{round_num}/pitstops", limit=100
                )
                
                for ps in pitstop_data:
                    for stop in ps.get("PitStops", []):
                        row = {
                            "season": int(season),
                            "round": int(round_num),
                            "raceName": race_name,
                            "driverId": stop.get("driverId", ""),
                            "lap": int(stop.get("lap", 0)),
                            "stop": int(stop.get("stop", 0)),
                            "time": stop.get("time", ""),
                            "duration": stop.get("duration", ""),
                        }
                        all_rows.append(row)
            except Exception as e:
                logger.debug(f"    Pitstops {season} R{round_num}: {e}")
        
        season_stops = len([r for r in all_rows if r["season"] == season])
        logger.info(f"    {season}: {season_stops} pit stops")
    
    return pd.DataFrame(all_rows)


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_extraction(seasons: list = None, config_path: str = "configs/config.yaml"):
    """Run the full Jolpica extraction pipeline."""
    config = load_config(config_path)
    jolpica_cfg = config["extraction"]["jolpica"]
    
    base_url = jolpica_cfg["base_url"]
    if seasons is None:
        seasons = jolpica_cfg["seasons"]
    
    output_dir = Path(config["paths"]["raw"]["jolpica"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Jolpica extraction: seasons={seasons}")
    logger.info(f"Output directory: {output_dir}")
    start = time.time()
    
    # 1. Race results
    results_df = extract_results(base_url, seasons)
    results_df.to_parquet(output_dir / "results.parquet", index=False)
    logger.info(f"  ✓ results.parquet ({len(results_df):,} rows)")
    
    # 2. Qualifying
    qual_df = extract_qualifying(base_url, seasons)
    qual_df.to_parquet(output_dir / "qualifying.parquet", index=False)
    logger.info(f"  ✓ qualifying.parquet ({len(qual_df):,} rows)")
    
    # 3. Constructor standings
    standings_df = extract_constructor_standings(base_url, seasons)
    standings_df.to_parquet(output_dir / "constructor_standings.parquet", index=False)
    logger.info(f"  ✓ constructor_standings.parquet ({len(standings_df):,} rows)")
    
    # 4. Pit stops
    pitstops_df = extract_pitstops(base_url, seasons, results_df=results_df)
    pitstops_df.to_parquet(output_dir / "pitstops.parquet", index=False)
    logger.info(f"  ✓ pitstops.parquet ({len(pitstops_df):,} rows)")
    
    elapsed = time.time() - start
    logger.info(f"\n  Jolpica extraction complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Extract historical F1 data from Jolpica API.")
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_extraction(seasons=args.seasons, config_path=args.config)


if __name__ == "__main__":
    main()
