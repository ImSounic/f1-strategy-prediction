"""
Backtesting: Predicted vs Actual Race Strategies
=================================================
Enriched validation that goes beyond stop-count matching:

  1. Compound sequence match (exact compound order)
  2. Pit lap accuracy (±N laps from actual pit laps)
  3. Rank of actual strategy in model's predicted rankings
  4. Per-circuit performance breakdown

Uses the existing validation_rolling_report.json (Fold 3: 2022-2024 → 2025)
and enriches it with data from precomputed strategy results + actual pit laps.

Output:
    frontend/src/data/backtestResults.ts

Usage:
    python -m src.analysis.backtest
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Compound name normalizer ──

COMPOUND_MAP = {
    "SOFT": "S", "MEDIUM": "M", "HARD": "H",
    "INTERMEDIATE": "I", "WET": "W",
    "C1": "H", "C2": "H", "C3": "M", "C4": "M", "C5": "S", "C6": "S",
}


def normalize_compound(c: str) -> str:
    return COMPOUND_MAP.get(c.upper().strip(), c[0].upper() if c else "?")


def normalize_sequence(compounds: list[str]) -> str:
    """Convert ['SOFT', 'MEDIUM', 'HARD'] → 'S-M-H'"""
    return "-".join(normalize_compound(c) for c in compounds)


def parse_strategy_compounds(strategy_name: str) -> list[str]:
    """Parse '2-stop HARD→MEDIUM→SOFT (18/18/21)' → ['HARD', 'MEDIUM', 'SOFT']"""
    if "→" in strategy_name:
        parts = strategy_name.split("(")[0].strip()
        parts = parts.split(" ", 1)[1] if " " in parts else parts
        return [c.strip() for c in parts.split("→")]
    if "→" in strategy_name:
        parts = strategy_name.split("(")[0].strip()
        parts = parts.split(" ", 1)[1] if " " in parts else parts
        return [c.strip() for c in parts.split("→")]
    return []


def parse_strategy_stint_lengths(strategy_name: str) -> list[int]:
    """Parse '2-stop HARD→MEDIUM→SOFT (18/18/21)' → [18, 18, 21]"""
    if "(" in strategy_name and ")" in strategy_name:
        inner = strategy_name.split("(")[1].split(")")[0]
        try:
            return [int(x) for x in inner.split("/")]
        except ValueError:
            return []
    return []


def compute_pit_laps(stint_lengths: list[int]) -> list[int]:
    """Convert stint lengths to pit lap numbers. [18, 18, 21] → [18, 36]"""
    pit_laps = []
    cumulative = 0
    for length in stint_lengths[:-1]:
        cumulative += length
        pit_laps.append(cumulative)
    return pit_laps


def find_strategy_rank(rankings: list[dict], actual_compounds: list[str], actual_stops: int) -> dict:
    """Find where the actual strategy ranks in our predictions.

    Returns:
        dict with rank info: exact compound match rank, stop count match rank,
        and closest compound match with rank.
    """
    actual_norm = normalize_sequence(actual_compounds)
    actual_set = set(normalize_compound(c) for c in actual_compounds)

    result = {
        "exact_compound_rank": None,
        "stop_count_rank": None,
        "closest_match_rank": None,
        "closest_match_name": None,
        "closest_match_score": 0,
    }

    for i, r in enumerate(rankings):
        strat_compounds = parse_strategy_compounds(r.get("strategy_name", r.get("name", "")))
        strat_norm = normalize_sequence(strat_compounds)
        strat_stops = r.get("num_stops", r.get("stops", 0))

        # Exact compound sequence match
        if strat_norm == actual_norm and result["exact_compound_rank"] is None:
            result["exact_compound_rank"] = i + 1

        # Stop count match (first occurrence)
        if strat_stops == actual_stops and result["stop_count_rank"] is None:
            result["stop_count_rank"] = i + 1

        # Closest compound match: score based on shared compounds + same count
        strat_set = set(normalize_compound(c) for c in strat_compounds)
        shared = len(actual_set & strat_set)
        same_stops = 1 if strat_stops == actual_stops else 0
        same_len = 1 if len(strat_compounds) == len(actual_compounds) else 0
        score = shared * 2 + same_stops * 3 + same_len * 1

        # Bonus for matching order
        if strat_norm == actual_norm:
            score += 10

        if score > result["closest_match_score"]:
            result["closest_match_score"] = score
            result["closest_match_rank"] = i + 1
            result["closest_match_name"] = r.get("strategy_name", r.get("name", ""))

    return result


def compute_pit_lap_error(predicted_stint_lengths: list[int], actual_pit_laps: list[int]) -> dict:
    """Compute pit lap prediction accuracy."""
    predicted_pits = compute_pit_laps(predicted_stint_lengths)

    if not predicted_pits or not actual_pit_laps:
        return {"mean_error": None, "max_error": None, "pit_lap_errors": []}

    # Match predicted pits to actual pits (greedy closest)
    errors = []
    used = set()
    for p_lap in predicted_pits:
        best_err = float("inf")
        best_idx = -1
        for j, a_lap in enumerate(actual_pit_laps):
            if j not in used and abs(p_lap - a_lap) < best_err:
                best_err = abs(p_lap - a_lap)
                best_idx = j
        if best_idx >= 0:
            used.add(best_idx)
            errors.append({
                "predicted": p_lap,
                "actual": actual_pit_laps[best_idx],
                "error": best_err,
            })

    mean_err = sum(e["error"] for e in errors) / len(errors) if errors else None
    max_err = max(e["error"] for e in errors) if errors else None

    return {"mean_error": mean_err, "max_error": max_err, "pit_lap_errors": errors}


def load_actual_pit_laps(pitstops_path: Path, results_path: Path, season: int) -> dict:
    """Load actual pit lap numbers for race winners."""
    pitstops = pd.read_parquet(pitstops_path)
    results = pd.read_parquet(results_path)

    winners = results[(results["season"] == season) & (results["position"] == 1)]
    pit_laps = {}

    for _, winner in winners.iterrows():
        rnd = int(winner["round"])
        driver_id = winner["driverId"]

        pits = pitstops[
            (pitstops["season"] == season)
            & (pitstops["round"] == rnd)
            & (pitstops["driverId"] == driver_id)
        ].sort_values("lap")

        pit_laps[(season, rnd)] = pits["lap"].tolist()

    return pit_laps


def load_precomputed_rankings(results_dir: Path, circuit_key: str, season: int) -> list[dict] | None:
    """Load precomputed MC strategy rankings for a circuit."""
    path = results_dir / f"strategy_{circuit_key}_{season}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("rankings", [])


def circuit_key_from_name(circuit_name: str, circuits_csv: Path) -> str | None:
    """Look up circuit_key from circuit_name."""
    circuits = pd.read_csv(circuits_csv)
    row = circuits[circuits["circuit_name"] == circuit_name]
    if not row.empty:
        return row.iloc[0]["circuit_key"]
    return None


def run_backtest(config_path: str = "configs/config.yaml"):
    config_yaml = yaml.safe_load(open(config_path))
    raw_paths = config_yaml["paths"]["raw"]

    pitstops_path = Path(raw_paths["jolpica"]) / "pitstops.parquet"
    results_path = Path(raw_paths["jolpica"]) / "results.parquet"
    circuits_csv = Path(raw_paths["supplementary"]) / "pirelli_circuit_characteristics.csv"
    results_dir = Path("results")
    validation_path = results_dir / "validation_rolling_report.json"

    if not validation_path.exists():
        logger.error("validation_rolling_report.json not found. Run strategy_validation_rolling first.")
        return

    with open(validation_path) as f:
        val_report = json.load(f)

    # Load all fold data
    all_backtest_races = []
    season_summaries = []

    for fold in val_report["folds"]:
        val_season = fold["val_season"]
        train_seasons = fold["train_seasons"]

        logger.info(f"\n{'─' * 70}")
        logger.info(f"  FOLD: Train {train_seasons} → Backtest {val_season}")
        logger.info(f"{'─' * 70}")

        # Load actual pit laps for this season
        actual_pit_laps = load_actual_pit_laps(pitstops_path, results_path, val_season)

        # Load circuits CSV for key lookup
        circuits = pd.read_csv(circuits_csv)

        # Counters
        total_dry = 0
        stops_match = 0
        compound_match = 0
        compound_partial = 0
        ranks_in_top3 = 0
        ranks_in_top5 = 0
        pit_errors = []

        for race in fold["races"]:
            circuit_name = race["circuit"]
            is_wet = race["is_wet"]

            # Find circuit_key
            circuit_row = circuits[circuits["circuit_name"] == circuit_name]
            if circuit_row.empty:
                logger.warning(f"  Skipping {circuit_name}: not in circuits CSV")
                continue
            circuit_key = circuit_row.iloc[0]["circuit_key"]

            # Find round number
            results_df = pd.read_parquet(results_path)
            rnd_row = results_df[
                (results_df["season"] == val_season)
                & (results_df["circuitName"].str.contains(circuit_name.replace(" Grand Prix", "").strip(), case=False, na=False))
                & (results_df["position"] == 1)
            ]
            if rnd_row.empty:
                # Try matching by raceName
                rnd_row = results_df[
                    (results_df["season"] == val_season)
                    & (results_df["raceName"] == circuit_name)
                    & (results_df["position"] == 1)
                ]
            rnd = int(rnd_row.iloc[0]["round"]) if not rnd_row.empty else None

            # Get actual compounds and pit laps
            actual_compounds_raw = race.get("actual_compounds", "").replace(" → ", "→").split("→")
            actual_compounds_raw = [c.strip() for c in actual_compounds_raw if c.strip()]
            actual_stops = race["actual_stops"]
            actual_pits = actual_pit_laps.get((val_season, rnd), []) if rnd else []

            # Get our predicted strategy
            recommended = race["recommended_strategy"]
            recommended_stops = race["recommended_stops"]
            recommended_compounds = parse_strategy_compounds(recommended)
            recommended_stints = parse_strategy_stint_lengths(recommended)
            recommended_pits = compute_pit_laps(recommended_stints)

            # Load precomputed rankings for richer analysis
            rankings = load_precomputed_rankings(results_dir, circuit_key, val_season)

            # ── Metrics ──

            # 1. Compound sequence match
            actual_norm = normalize_sequence(actual_compounds_raw)
            recommended_norm = normalize_sequence(recommended_compounds)
            exact_compound = actual_norm == recommended_norm

            # Partial match: same compounds used (ignoring order)
            actual_set = set(normalize_compound(c) for c in actual_compounds_raw)
            rec_set = set(normalize_compound(c) for c in recommended_compounds)
            compounds_shared = actual_set & rec_set
            partial_compound = len(compounds_shared) >= min(len(actual_set), len(rec_set)) and len(compounds_shared) > 0

            # 2. Pit lap accuracy
            pit_result = compute_pit_lap_error(recommended_stints, actual_pits)

            # 3. Strategy rank in predictions
            rank_info = {"exact_compound_rank": None, "stop_count_rank": None, "closest_match_rank": None, "closest_match_name": None}
            if rankings and not is_wet:
                rank_info = find_strategy_rank(rankings, actual_compounds_raw, actual_stops)

            # ── Aggregate (dry only) ──
            if not is_wet:
                total_dry += 1
                if recommended_stops == actual_stops:
                    stops_match += 1
                if exact_compound:
                    compound_match += 1
                if partial_compound:
                    compound_partial += 1
                if rank_info["stop_count_rank"] and rank_info["stop_count_rank"] <= 3:
                    ranks_in_top3 += 1
                if rank_info["stop_count_rank"] and rank_info["stop_count_rank"] <= 5:
                    ranks_in_top5 += 1
                if pit_result["mean_error"] is not None:
                    pit_errors.append(pit_result["mean_error"])

            # Verdict
            if is_wet:
                verdict = "WET"
            elif exact_compound:
                verdict = "EXACT"
            elif recommended_stops == actual_stops and partial_compound:
                verdict = "PARTIAL"
            elif recommended_stops == actual_stops:
                verdict = "STOPS_ONLY"
            else:
                verdict = "MISS"

            marker = {"EXACT": "★", "PARTIAL": "◐", "STOPS_ONLY": "✓", "MISS": "✗", "WET": "☁"}[verdict]

            logger.info(
                f"  {marker} {circuit_name:<28} "
                f"Actual: {actual_norm:<12} "
                f"Ours: {recommended_norm:<12} "
                f"PitErr: {pit_result['mean_error']:>5.1f} laps" if pit_result["mean_error"] is not None else
                f"  {marker} {circuit_name:<28} "
                f"Actual: {actual_norm:<12} "
                f"Ours: {recommended_norm:<12} "
                f"PitErr: N/A"
            )

            backtest_entry = {
                "season": val_season,
                "round": rnd,
                "circuit": circuit_name,
                "circuitKey": circuit_key,
                "winner": race["winner"],
                "isWet": is_wet,
                "verdict": verdict,
                # Actual
                "actualStops": actual_stops,
                "actualCompounds": actual_compounds_raw,
                "actualCompoundNorm": actual_norm,
                "actualPitLaps": actual_pits,
                # Predicted
                "predictedStrategy": recommended,
                "predictedStops": recommended_stops,
                "predictedCompounds": recommended_compounds,
                "predictedCompoundNorm": recommended_norm,
                "predictedPitLaps": recommended_pits,
                # Metrics
                "stopsMatch": recommended_stops == actual_stops,
                "compoundExactMatch": exact_compound,
                "compoundPartialMatch": partial_compound,
                "pitLapMeanError": round(pit_result["mean_error"], 1) if pit_result["mean_error"] is not None else None,
                "pitLapMaxError": round(pit_result["max_error"], 1) if pit_result["max_error"] is not None else None,
                "pitLapErrors": pit_result["pit_lap_errors"],
                # Ranking
                "actualStrategyRankByStops": rank_info["stop_count_rank"],
                "actualStrategyRankByCompounds": rank_info["exact_compound_rank"],
                "closestMatchRank": rank_info["closest_match_rank"],
                "closestMatchName": rank_info["closest_match_name"],
            }
            all_backtest_races.append(backtest_entry)

        # Season summary
        avg_pit_err = round(sum(pit_errors) / len(pit_errors), 1) if pit_errors else None
        season_summaries.append({
            "season": val_season,
            "trainSeasons": train_seasons,
            "totalRaces": len(fold["races"]),
            "dryRaces": total_dry,
            "stopsMatchRate": round(stops_match / max(total_dry, 1) * 100, 1),
            "compoundExactRate": round(compound_match / max(total_dry, 1) * 100, 1),
            "compoundPartialRate": round(compound_partial / max(total_dry, 1) * 100, 1),
            "top3Rate": round(ranks_in_top3 / max(total_dry, 1) * 100, 1),
            "top5Rate": round(ranks_in_top5 / max(total_dry, 1) * 100, 1),
            "avgPitLapError": avg_pit_err,
        })

        logger.info(f"\n  Season {val_season} Summary (dry only):")
        logger.info(f"    Stops match:         {stops_match}/{total_dry} ({stops_match/max(total_dry,1)*100:.0f}%)")
        logger.info(f"    Compound exact:      {compound_match}/{total_dry} ({compound_match/max(total_dry,1)*100:.0f}%)")
        logger.info(f"    Compound partial:    {compound_partial}/{total_dry} ({compound_partial/max(total_dry,1)*100:.0f}%)")
        logger.info(f"    Avg pit lap error:   {avg_pit_err} laps")

    # ── Compute overall summary ──
    dry_races = [r for r in all_backtest_races if not r["isWet"]]
    overall = {
        "totalRaces": len(all_backtest_races),
        "dryRaces": len(dry_races),
        "wetRaces": len(all_backtest_races) - len(dry_races),
        "stopsMatchRate": round(sum(1 for r in dry_races if r["stopsMatch"]) / max(len(dry_races), 1) * 100, 1),
        "compoundExactRate": round(sum(1 for r in dry_races if r["compoundExactMatch"]) / max(len(dry_races), 1) * 100, 1),
        "compoundPartialRate": round(sum(1 for r in dry_races if r["compoundPartialMatch"]) / max(len(dry_races), 1) * 100, 1),
        "verdictCounts": {
            "EXACT": sum(1 for r in all_backtest_races if r["verdict"] == "EXACT"),
            "PARTIAL": sum(1 for r in all_backtest_races if r["verdict"] == "PARTIAL"),
            "STOPS_ONLY": sum(1 for r in all_backtest_races if r["verdict"] == "STOPS_ONLY"),
            "MISS": sum(1 for r in all_backtest_races if r["verdict"] == "MISS"),
            "WET": sum(1 for r in all_backtest_races if r["verdict"] == "WET"),
        },
    }

    # Compute avg pit lap error across all dry races with data
    all_pit_errors = [r["pitLapMeanError"] for r in dry_races if r["pitLapMeanError"] is not None]
    overall["avgPitLapError"] = round(sum(all_pit_errors) / len(all_pit_errors), 1) if all_pit_errors else None

    # ── Generate TypeScript ──
    generate_typescript(all_backtest_races, season_summaries, overall)

    # ── Save JSON too ──
    output = {
        "overall": overall,
        "seasonSummaries": season_summaries,
        "races": all_backtest_races,
    }
    with open(results_dir / "backtest_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  ✓ JSON saved: results/backtest_report.json")


def generate_typescript(races: list[dict], summaries: list[dict], overall: dict):
    """Generate TypeScript data file for frontend."""
    ts_path = Path("frontend/src/data/backtestResults.ts")

    lines = []
    lines.append("// Auto-generated by src/analysis/backtest.py — do not edit manually")
    lines.append("")
    lines.append("export interface PitLapError {")
    lines.append("  predicted: number;")
    lines.append("  actual: number;")
    lines.append("  error: number;")
    lines.append("}")
    lines.append("")
    lines.append("export interface BacktestRace {")
    lines.append("  season: number;")
    lines.append("  round: number | null;")
    lines.append("  circuit: string;")
    lines.append("  circuitKey: string;")
    lines.append("  winner: string;")
    lines.append("  isWet: boolean;")
    lines.append("  verdict: 'EXACT' | 'PARTIAL' | 'STOPS_ONLY' | 'MISS' | 'WET';")
    lines.append("  actualStops: number;")
    lines.append("  actualCompounds: string[];")
    lines.append("  actualCompoundNorm: string;")
    lines.append("  actualPitLaps: number[];")
    lines.append("  predictedStrategy: string;")
    lines.append("  predictedStops: number;")
    lines.append("  predictedCompounds: string[];")
    lines.append("  predictedCompoundNorm: string;")
    lines.append("  predictedPitLaps: number[];")
    lines.append("  stopsMatch: boolean;")
    lines.append("  compoundExactMatch: boolean;")
    lines.append("  compoundPartialMatch: boolean;")
    lines.append("  pitLapMeanError: number | null;")
    lines.append("  pitLapMaxError: number | null;")
    lines.append("  pitLapErrors: PitLapError[];")
    lines.append("  actualStrategyRankByStops: number | null;")
    lines.append("  actualStrategyRankByCompounds: number | null;")
    lines.append("  closestMatchRank: number | null;")
    lines.append("  closestMatchName: string | null;")
    lines.append("}")
    lines.append("")
    lines.append("export interface SeasonSummary {")
    lines.append("  season: number;")
    lines.append("  trainSeasons: number[];")
    lines.append("  totalRaces: number;")
    lines.append("  dryRaces: number;")
    lines.append("  stopsMatchRate: number;")
    lines.append("  compoundExactRate: number;")
    lines.append("  compoundPartialRate: number;")
    lines.append("  top3Rate: number;")
    lines.append("  top5Rate: number;")
    lines.append("  avgPitLapError: number | null;")
    lines.append("}")
    lines.append("")
    lines.append("export interface BacktestOverall {")
    lines.append("  totalRaces: number;")
    lines.append("  dryRaces: number;")
    lines.append("  wetRaces: number;")
    lines.append("  stopsMatchRate: number;")
    lines.append("  compoundExactRate: number;")
    lines.append("  compoundPartialRate: number;")
    lines.append("  avgPitLapError: number | null;")
    lines.append("  verdictCounts: {")
    lines.append("    EXACT: number;")
    lines.append("    PARTIAL: number;")
    lines.append("    STOPS_ONLY: number;")
    lines.append("    MISS: number;")
    lines.append("    WET: number;")
    lines.append("  };")
    lines.append("}")
    lines.append("")
    lines.append("export interface BacktestData {")
    lines.append("  overall: BacktestOverall;")
    lines.append("  seasonSummaries: SeasonSummary[];")
    lines.append("  races: BacktestRace[];")
    lines.append("}")
    lines.append("")

    # Serialize data
    lines.append(f"export const backtestData: BacktestData = {json.dumps({'overall': overall, 'seasonSummaries': summaries, 'races': races}, indent=2)};")
    lines.append("")

    ts_path.write_text("\n".join(lines))
    logger.info(f"  ✓ TypeScript saved: {ts_path}")


if __name__ == "__main__":
    run_backtest()
