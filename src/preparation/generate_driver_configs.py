"""
Driver config generator
========================
Reproducibly derives per-season driver ratings from committed Jolpica/feature
data, matching the documented 2024/25 methodology. Writes to configs/generated/
by default so output can be reviewed before replacing committed files.

Usage:
    python -m src.preparation.generate_driver_configs --seasons 2022 2023 2024 2025
    python -m src.preparation.generate_driver_configs --seasons 2026 --out-dir configs
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.preparation.driver_config_helpers import (
    best_quali_time, minmax_normalize, constructor_to_team, driver_name_from_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MIN_STINT_LAPS = 5

# Canonical team metadata + circuit overtaking difficulty (circuit constant),
# sourced from the committed 2025 config. Used for all seasons.
TEAMS = {
    "mclaren": {"color": "#FF8000", "name": "McLaren"},
    "red_bull": {"color": "#3671C6", "name": "Red Bull Racing"},
    "ferrari": {"color": "#E8002D", "name": "Ferrari"},
    "mercedes": {"color": "#27F4D2", "name": "Mercedes"},
    "aston_martin": {"color": "#229971", "name": "Aston Martin"},
    "alpine": {"color": "#FF87BC", "name": "Alpine"},
    "williams": {"color": "#64C4FF", "name": "Williams"},
    "rb": {"color": "#6692FF", "name": "Racing Bulls"},
    "sauber": {"color": "#52E252", "name": "Kick Sauber"},
    "haas": {"color": "#B6BABD", "name": "Haas"},
}

CIRCUIT_OVERTAKING_DIFFICULTY = {
    "bahrain": 0.70, "jeddah": 0.75, "albert_park": 0.55, "suzuka": 0.40,
    "shanghai": 0.65, "miami": 0.60, "imola": 0.35, "monaco": 0.10,
    "montreal": 0.55, "barcelona": 0.35, "spielberg": 0.55, "silverstone": 0.55,
    "hungaroring": 0.30, "spa": 0.65, "zandvoort": 0.25, "monza": 0.80,
    "baku": 0.65, "singapore": 0.30, "cota": 0.60, "mexico": 0.55,
    "interlagos": 0.65, "las_vegas": 0.70, "lusail": 0.55, "yas_marina": 0.55,
}


def _pace_delta(qual: pd.DataFrame) -> dict:
    rows = []
    for (rnd, drv), g in qual.groupby(["round", "driverCode"]):
        r0 = g.iloc[0]
        t = best_quali_time(r0.get("Q1"), r0.get("Q2"), r0.get("Q3"))
        if t is not None:
            rows.append((rnd, drv, t))
    if not rows:
        return {}
    bt = pd.DataFrame(rows, columns=["round", "driver", "t"])
    pole = bt.groupby("round")["t"].min().rename("pole")
    bt = bt.join(pole, on="round")
    bt["gap"] = bt["t"] - bt["pole"]
    med = bt.groupby("driver")["gap"].median()
    med = med - med.min()
    return {d: round(float(v), 2) for d, v in med.items()}


def _overtaking(res: pd.DataFrame) -> dict:
    df = res[res["position"].notna()].copy()
    df["grid_eff"] = df["grid"].where(df["grid"] > 0, 20)
    df["gained"] = df["grid_eff"] - df["position"]
    mean_gain = df.groupby("driverCode")["gained"].mean()
    drivers = list(mean_gain.index)
    vals = minmax_normalize(mean_gain.tolist(), 0.40, 0.95)
    return {d: round(float(v), 2) for d, v in zip(drivers, vals)}


def _tyre_management(stints: pd.DataFrame) -> dict:
    df = stints[stints["StintLength"] >= MIN_STINT_LAPS].copy()
    comp_med = df.groupby("Compound")["StintLength"].median()
    df["ratio"] = df["StintLength"] / df["Compound"].map(comp_med)
    med = df.groupby("Driver")["ratio"].median()
    drivers = list(med.index)
    vals = minmax_normalize(med.tolist(), 0.50, 0.95)
    return {d: round(float(v), 2) for d, v in zip(drivers, vals)}


def _teams_and_mates(res: pd.DataFrame) -> tuple:
    team_of, name_of = {}, {}
    for drv, g in res.groupby("driverCode"):
        team_of[drv] = constructor_to_team(g["constructorId"].mode().iloc[0])
        name_of[drv] = driver_name_from_id(g["driverId"].mode().iloc[0])
    by_team = {}
    for drv, tk in team_of.items():
        by_team.setdefault(tk, []).append(drv)
    rounds_of = {drv: set(g["round"]) for drv, g in res.groupby("driverCode")}
    mates = {}
    for drv, tk in team_of.items():
        best, best_ov = "", -1
        for other in by_team[tk]:
            if other == drv:
                continue
            ov = len(rounds_of[drv] & rounds_of[other])
            if ov > best_ov:
                best_ov, best = ov, other
        mates[drv] = best
    return team_of, mates, name_of


def generate_season(season: int, raw_dir: Path, features_dir: Path) -> dict:
    qual = pd.read_parquet(raw_dir / "qualifying.parquet")
    res = pd.read_parquet(raw_dir / "results.parquet")
    stints = pd.read_parquet(features_dir / "stint_features.parquet")

    qual = qual[qual["season"] == season]
    res = res[res["season"] == season]
    stints = stints[stints["Season"] == season]

    pace = _pace_delta(qual)
    overt = _overtaking(res)
    tyre = _tyre_management(stints)
    team_of, mates, name_of = _teams_and_mates(res)

    pace_fallback = (max(pace.values()) + 0.1) if pace else 0.0
    drivers = []
    for drv in sorted(team_of.keys()):
        drivers.append({
            "code": drv,
            "name": name_of.get(drv, drv),
            "team": team_of[drv],
            "pace_delta": pace.get(drv, round(pace_fallback, 2)),
            "overtaking": overt.get(drv, 0.40),
            "tyre_management": tyre.get(drv, 0.70),
            "teammate": mates.get(drv, ""),
        })
    drivers.sort(key=lambda d: d["pace_delta"])
    reference = drivers[0]["code"] if drivers else ""

    return {
        "season": season,
        "reference_driver": reference,
        "notes": (f"Auto-generated by generate_driver_configs.py from "
                  f"Jolpica/feature data for {season}. pace_delta = median "
                  f"qualifying gap-to-pole (Q3>Q2>Q1), referenced to fastest "
                  f"driver. overtaking = mean positions gained, normalized "
                  f"0.40-0.95. tyre_management = median stint-length ratio vs "
                  f"compound median, normalized 0.50-0.95. Team keys "
                  f"canonicalized (alfa->sauber, alphatauri->rb); names "
                  f"best-effort from driverId."),
        "data_sources": {
            "pace_delta": "Median per-round qualifying gap to pole (Q3>Q2>Q1)",
            "overtaking": "Mean positions gained per race, normalized 0.40-0.95",
            "tyre_management": "Median stint-length ratio vs compound median, normalized 0.50-0.95",
        },
        "teams": TEAMS,
        "drivers": drivers,
        "circuit_overtaking_difficulty": CIRCUIT_OVERTAKING_DIFFICULTY,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate driver config JSONs")
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw/jolpica")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--out-dir", type=str, default="configs/generated")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        cfg = generate_season(season, raw_dir, features_dir)
        out_path = out_dir / f"drivers_{season}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"  wrote {out_path}  ({len(cfg['drivers'])} drivers, ref={cfg['reference_driver']})")


if __name__ == "__main__":
    main()
