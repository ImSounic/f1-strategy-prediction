"""
Add circuit-characteristics rows for a new season
=================================================
The Pirelli circuit-characteristics CSV is manually maintained and the modeling
pipeline maps FastF1 compound *names* to C-codes using its per-race allocation
columns. A new season therefore needs rows in this CSV before clean_laps /
feature_engineering can process it.

F1 circuits are physically stable year-to-year, so we generate a new season's
rows by copying each round's circuit characteristics from that circuit's latest
prior-year row (changing only season + round_number).

Caveat: the per-race compound *allocation* (hard/medium/soft C-codes) is copied
from the prior year as an approximation. It still round-trips name<->code so the
SOFT/MEDIUM/HARD logic is correct everywhere, and the degradation model is
compound-insensitive (compound realism comes from the regulation-profile deg
multiplier calibrated on the new season). Replace with the real allocation if a
precise per-race C-code mapping is needed.

Usage:
    python -m src.preparation.add_season_circuits --season 2026
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CSV = Path("data/raw/supplementary/pirelli_circuit_characteristics.csv")

# Known calendars: round_number -> circuit_key (matching existing circuit_key
# naming). Extend as new rounds are completed / new seasons are added.
SCHEDULES = {
    2026: {
        1: "albert_park",   # Australian GP
        2: "shanghai",      # Chinese GP
        3: "suzuka",        # Japanese GP
        4: "miami",         # Miami GP
        5: "montreal",      # Canadian GP
        6: "monaco",        # Monaco GP
        7: "barcelona",     # Barcelona GP
        8: "spielberg",     # Austrian GP
    },
}


def add_season(season: int) -> None:
    df = pd.read_csv(CSV)
    sched = SCHEDULES.get(season)
    if not sched:
        raise SystemExit(f"No schedule for {season}; add it to SCHEDULES in this file.")

    df = df[df["season"] != season]  # idempotent: drop any prior copy

    new_rows = []
    for rnd, ckey in sorted(sched.items()):
        prior = df[df["circuit_key"] == ckey].sort_values("season")
        if prior.empty:
            logger.warning(f"  no prior row for circuit_key '{ckey}' — skipping round {rnd}")
            continue
        row = prior.iloc[-1].copy()
        row["season"] = season
        row["round_number"] = rnd
        new_rows.append(row)
        logger.info(f"  round {rnd:>2} {ckey:<12} <- copied from {int(prior.iloc[-1]['season'])}")

    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["season", "round_number"]).reset_index(drop=True)
    out.to_csv(CSV, index=False)
    logger.info(f"Wrote {len(new_rows)} {season} circuit rows -> {CSV} ({len(out)} total)")


def main():
    parser = argparse.ArgumentParser(description="Add a season's circuit rows to the Pirelli CSV")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    add_season(args.season)


if __name__ == "__main__":
    main()
