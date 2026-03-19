"""
Compound Strategy Prior Model
===============================
Learns historical compound sequence probabilities from real race data
and provides likelihood scores for strategy reranking.

Key insight: the MC simulator's noise (std ~145s) is 6x larger than
strategy time differences (~24s), making pure time-based ranking unreliable
for compound selection. This prior injects real-world compound patterns
to break ties.

The prior captures:
  1. Starting compound likelihood (by circuit category)
  2. Compound transition probabilities (1st → 2nd, 2nd → 3rd)
  3. Stop count distribution per circuit category
  4. Circuit-specific overrides where data exists

Usage:
    prior = CompoundPrior.from_data(features_dir, circuits_csv)
    score = prior.score_strategy("MEDIUM-HARD", circuit_key, n_stops=1)
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CompoundPrior:
    """Historical compound sequence probabilities."""

    # P(starting_compound | circuit_category)
    start_probs: dict[str, dict[str, float]] = field(default_factory=dict)

    # P(next_compound | current_compound) — global transitions
    transition_probs: dict[str, dict[str, float]] = field(default_factory=dict)

    # P(n_stops | circuit_category) — stop count distribution
    stop_probs: dict[str, dict[int, float]] = field(default_factory=dict)

    # Full strategy sequence counts per circuit (when enough data)
    circuit_strategy_counts: dict[str, Counter] = field(default_factory=dict)

    # Circuit → category mapping
    circuit_categories: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        features_dir: str | Path,
        circuits_csv: str | Path,
        results_path: str | Path,
        min_seasons: int = 2,
    ) -> "CompoundPrior":
        """Build prior from historical stint and results data."""
        features_dir = Path(features_dir)
        stints = pd.read_parquet(features_dir / "stint_features.parquet")
        circuits = pd.read_csv(circuits_csv)
        results = pd.read_parquet(results_path)

        # Map C1-C6 to SOFT/MEDIUM/HARD using circuit allocation
        merged = stints.merge(
            circuits[["season", "round_number", "circuit_key",
                       "hard_compound", "medium_compound", "soft_compound",
                       "tyre_stress", "asphalt_abrasiveness", "pit_loss_seconds",
                       "total_laps"]],
            left_on=["Season", "RoundNumber"],
            right_on=["season", "round_number"],
            how="inner",
        )

        def _map_compound(row):
            c = row["Compound"]
            if c == row["soft_compound"]:
                return "SOFT"
            elif c == row["medium_compound"]:
                return "MEDIUM"
            elif c == row["hard_compound"]:
                return "HARD"
            return c  # INTERMEDIATE, WET, etc.

        merged["CompoundName"] = merged.apply(_map_compound, axis=1)

        # Filter to dry compounds only
        dry = merged[merged["CompoundName"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

        # ── Categorize circuits by tyre stress ──
        circuit_stress = (
            circuits.groupby("circuit_key")["tyre_stress"]
            .mean()
            .to_dict()
        )

        def _categorize(circuit_key: str) -> str:
            stress = circuit_stress.get(circuit_key, 50)
            if stress < 35:
                return "low_stress"
            elif stress < 55:
                return "med_stress"
            else:
                return "high_stress"

        circuit_cats = {ck: _categorize(ck) for ck in circuit_stress}

        # ── Build strategy sequences per driver-race ──
        sequences = (
            dry.sort_values(["Season", "RoundNumber", "Driver", "StintNumber"])
            .groupby(["Season", "RoundNumber", "Driver", "circuit_key"])
            .agg(
                Strategy=("CompoundName", lambda x: "-".join(x)),
                NumStints=("CompoundName", "count"),
                FirstCompound=("CompoundName", "first"),
            )
            .reset_index()
        )
        sequences["NumStops"] = sequences["NumStints"] - 1
        sequences["Category"] = sequences["circuit_key"].map(circuit_cats)

        # Merge with results for grid position (optional enrichment)
        sequences = sequences.merge(
            results[["season", "round", "driverCode", "grid", "position"]],
            left_on=["Season", "RoundNumber", "Driver"],
            right_on=["season", "round", "driverCode"],
            how="left",
        )

        # ── 1. Starting compound priors by circuit category ──
        start_probs = {}
        for cat in ["low_stress", "med_stress", "high_stress"]:
            subset = sequences[sequences["Category"] == cat]
            counts = subset["FirstCompound"].value_counts()
            total = counts.sum()
            if total == 0:
                start_probs[cat] = {"SOFT": 0.2, "MEDIUM": 0.6, "HARD": 0.2}
            else:
                start_probs[cat] = {
                    c: counts.get(c, 0) / total for c in ["SOFT", "MEDIUM", "HARD"]
                }

        # ── 2. Compound transitions ──
        stint_pairs = dry.sort_values(
            ["Season", "RoundNumber", "Driver", "StintNumber"]
        )
        transitions = defaultdict(Counter)
        for (season, rnd, driver), group in stint_pairs.groupby(
            ["Season", "RoundNumber", "Driver"]
        ):
            compounds = group.sort_values("StintNumber")["CompoundName"].tolist()
            for i in range(len(compounds) - 1):
                transitions[compounds[i]][compounds[i + 1]] += 1

        transition_probs = {}
        for from_c, to_counts in transitions.items():
            total = sum(to_counts.values())
            transition_probs[from_c] = {
                to_c: count / total for to_c, count in to_counts.items()
            }

        # ── 3. Stop count distribution by category ──
        stop_probs = {}
        for cat in ["low_stress", "med_stress", "high_stress"]:
            subset = sequences[sequences["Category"] == cat]
            counts = subset["NumStops"].value_counts()
            total = counts.sum()
            stop_probs[cat] = {
                int(stops): count / total
                for stops, count in counts.items()
            }

        # ── 4. Per-circuit strategy counts ──
        circuit_strat_counts = {}
        for ck, group in sequences.groupby("circuit_key"):
            n_seasons = group["Season"].nunique()
            if n_seasons >= min_seasons:
                circuit_strat_counts[ck] = Counter(group["Strategy"].tolist())

        prior = cls(
            start_probs=start_probs,
            transition_probs=transition_probs,
            stop_probs=stop_probs,
            circuit_strategy_counts=circuit_strat_counts,
            circuit_categories=circuit_cats,
        )

        n_sequences = len(sequences)
        n_circuits = len(circuit_strat_counts)
        logger.info(
            f"  Compound prior built: {n_sequences} driver-race sequences, "
            f"{n_circuits} circuits with per-circuit data"
        )

        return prior

    def score_strategy(
        self,
        strategy_sequence: str,
        circuit_key: str,
        n_stops: int,
    ) -> float:
        """Score a strategy by its historical likelihood.

        Args:
            strategy_sequence: e.g. "MEDIUM-HARD" or "SOFT-MEDIUM-HARD"
            circuit_key: e.g. "bahrain"
            n_stops: number of pit stops

        Returns:
            Log-likelihood score (higher = more likely in real data).
            Scores are comparable within a circuit.
        """
        compounds = strategy_sequence.split("-")
        if not compounds:
            return -10.0

        cat = self.circuit_categories.get(circuit_key, "med_stress")

        score = 0.0

        # ── Component 1: Starting compound (weight: 1.0) ──
        start_p = self.start_probs.get(cat, {}).get(compounds[0], 0.01)
        score += np.log(max(start_p, 0.01))

        # ── Component 2: Compound transitions (weight: 0.8 each) ──
        for i in range(len(compounds) - 1):
            trans_p = (
                self.transition_probs
                .get(compounds[i], {})
                .get(compounds[i + 1], 0.01)
            )
            score += 0.8 * np.log(max(trans_p, 0.01))

        # ── Component 3: Stop count (weight: 0.5) ──
        stop_p = self.stop_probs.get(cat, {}).get(n_stops, 0.05)
        score += 0.5 * np.log(max(stop_p, 0.01))

        # ── Component 4: Circuit-specific boost (weight: 1.5) ──
        if circuit_key in self.circuit_strategy_counts:
            circuit_counts = self.circuit_strategy_counts[circuit_key]
            total = sum(circuit_counts.values())
            circuit_p = circuit_counts.get(strategy_sequence, 0) / total
            if circuit_p > 0:
                score += 1.5 * np.log(circuit_p)
            else:
                # Small penalty for strategies never seen at this circuit
                score += 1.5 * np.log(0.005)

        return score

    def rerank_strategies(
        self,
        mc_results: list[dict],
        circuit_key: str,
        blend_weight: float = 0.3,
    ) -> list[dict]:
        """Rerank MC strategy results using compound prior.

        IMPORTANT: Only reranks within the same stop count tier.
        The MC simulator's stop count decision (1-stop vs 2-stop) is
        reliable (71% accuracy). The prior only influences compound
        ordering within each tier to fix the 24% → higher compound
        exact match rate.

        Args:
            mc_results: list of MC result dicts (sorted by median_time)
            circuit_key: circuit key
            blend_weight: prior influence (0-1), default 0.3

        Returns:
            Reranked list with added prior_score and blended_score fields.
        """
        if not mc_results:
            return mc_results

        # Compute prior scores for all strategies
        for result in mc_results:
            compounds = result.get("compound_sequence", "")
            seq = compounds.replace(" → ", "-").replace("→", "-")
            n_stops = result.get("num_stops", 0)
            result["prior_score"] = self.score_strategy(seq, circuit_key, n_stops)

        # Group by stop count
        from collections import defaultdict
        tiers = defaultdict(list)
        for i, result in enumerate(mc_results):
            result["_original_rank"] = i
            tiers[result.get("num_stops", 0)].append(result)

        # Rerank within each tier
        reranked = []
        for n_stops in sorted(tiers.keys()):
            tier = tiers[n_stops]

            if len(tier) <= 1:
                reranked.extend(tier)
                continue

            # Normalize MC rank within tier (0-1, best=1)
            for j, r in enumerate(tier):
                r["mc_rank_norm"] = 1.0 - (j / max(len(tier) - 1, 1))

            # Normalize prior scores within tier
            prior_scores = [r["prior_score"] for r in tier]
            min_p, max_p = min(prior_scores), max(prior_scores)
            range_p = max_p - min_p if max_p != min_p else 1.0
            for r in tier:
                r["prior_norm"] = (r["prior_score"] - min_p) / range_p

            # Blended score within tier
            for r in tier:
                r["blended_score"] = (
                    (1 - blend_weight) * r["mc_rank_norm"]
                    + blend_weight * r["prior_norm"]
                )

            tier.sort(key=lambda x: x["blended_score"], reverse=True)
            reranked.extend(tier)

        # Preserve MC's stop-count tier ordering strictly.
        # Within each tier, use the blended score (prior-adjusted).
        # The overall ordering between tiers is EXACTLY as MC had it:
        # if MC said 1-stop strategies are better overall, 1-stop stays on top.

        # Find which stop count MC ranked first, second, etc.
        mc_tier_order = []
        seen_tiers = set()
        for r in mc_results:
            ns = r.get("num_stops", 0)
            if ns not in seen_tiers:
                seen_tiers.add(ns)
                mc_tier_order.append(ns)

        # Build final list: iterate tiers in MC order, within each tier use blended order
        final = []
        tier_groups = {}
        for r in reranked:
            ns = r.get("num_stops", 0)
            if ns not in tier_groups:
                tier_groups[ns] = []
            tier_groups[ns].append(r)

        for ns in mc_tier_order:
            if ns in tier_groups:
                # Already sorted by blended_score desc within tier
                final.extend(tier_groups[ns])

        reranked = final

        # Update rank numbers
        for i, result in enumerate(reranked):
            result["rank"] = i + 1
            result.pop("_original_rank", None)

        return reranked
