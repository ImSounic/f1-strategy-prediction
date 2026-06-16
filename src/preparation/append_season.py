"""
Append one season's processed laps + features to the committed parquets
=======================================================================
Why this exists: only the *engineered* feature parquets are committed for past
seasons — the raw FastF1 laps are gitignored and not present on the cluster. So
we cannot re-run the normal full pipeline (it would regenerate features only for
whatever raw laps happen to be on disk, clobbering every other season).

This tool adds a single new season safely:
  1. back up the committed clean_laps + feature parquets,
  2. run clean_laps + feature_engineering scoped to the new season (these clobber
     the single-file parquets with season-only data),
  3. concat the new season onto the backed-up data (dropping any prior copy of
     that season so re-runs are idempotent) and write the combined parquets.

Prerequisite: the season's raw data must already be ingested
  - FastF1 laps/track_status/weather for the season
  - jolpica results/qualifying (re-fetched so they include the season)

Usage (run on a machine that has the raw season data, e.g. the HPC head node):
    python -m src.preparation.append_season --season 2026
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.preparation.clean_laps import run_cleaning
from src.preparation.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED = Path("data/processed/clean_laps.parquet")
FEATURES_DIR = Path("data/features")
FEATURE_FILES = [
    "stint_features.parquet",
    "lap_features.parquet",
    "driver_features.parquet",
    "team_features.parquet",
    "incident_features.parquet",
]


def _targets() -> list[Path]:
    return [PROCESSED] + [FEATURES_DIR / f for f in FEATURE_FILES]


def append_season(season: int, config_path: str = "configs/config.yaml") -> None:
    # 1. Back up whatever is committed (the multi-season baseline).
    backup = {}
    for f in _targets():
        if f.exists():
            backup[f] = pd.read_parquet(f)
            logger.info(f"  backed up {f.name}: {len(backup[f])} rows")

    # 2. Generate season-only processed laps + features (these clobber the files).
    logger.info(f"  generating season-only data for {season} ...")
    run_cleaning(seasons=[season], config_path=config_path)
    run_feature_engineering(seasons=[season], config_path=config_path)

    # 3. Merge: committed-minus-season + new-season, write combined.
    for f in _targets():
        if not f.exists():
            logger.warning(f"  {f.name}: season-only output missing, skipped")
            continue
        new = pd.read_parquet(f)
        if f in backup:
            old = backup[f]
            if "Season" in old.columns:
                old = old[old["Season"] != season]
            combined = pd.concat([old, new], ignore_index=True)
        else:
            combined = new
        combined.to_parquet(f, index=False)
        logger.info(f"  {f.name}: +{len(new)} ({season}) -> {len(combined)} total")

    logger.info(f"Done. Season {season} appended to committed parquets.")


def main():
    parser = argparse.ArgumentParser(description="Append a season to committed parquets")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    append_season(args.season, args.config)


if __name__ == "__main__":
    main()
