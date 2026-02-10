"""
Dynamic Time Warping (DTW) for Stint Similarity
==================================================
Uses DTW to find the most similar historical stint to a given
degradation pattern. This enables mid-race strategy adaptation:
"this stint looks like Verstappen's C3 stint at Silverstone 2023,
which degraded to a cliff at lap 25 — consider pitting early."

Also uses DTW to build a stint similarity matrix for clustering
degradation patterns beyond circuit-level aggregation.

Output:
    results/dtw_similarity_matrix.parquet
    results/dtw_example_matches.json
    results/dtw_dendrogram.png

Usage:
    python -m src.analysis.dtw_similarity
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two time series.
    
    DTW finds the optimal alignment between two sequences of different
    lengths, making it ideal for comparing stint degradation curves
    that may have different numbers of laps.
    
    Parameters:
        s1, s2: 1D arrays of fuel-corrected lap times
    
    Returns:
        DTW distance (lower = more similar)
    """
    n, m = len(s1), len(s2)
    
    # Cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1],    # match
            )
    
    return dtw_matrix[n, m]


def dtw_distance_normalized(s1: np.ndarray, s2: np.ndarray) -> float:
    """DTW distance normalized by path length."""
    n, m = len(s1), len(s2)
    dist = dtw_distance(s1, s2)
    return dist / (n + m)


def extract_stint_curves(features_dir: Path, min_laps: int = 8) -> pd.DataFrame:
    """Extract degradation curves (smoothed lap times) per stint."""
    logger.info("Extracting stint degradation curves...")
    
    laps = pd.read_parquet(features_dir / "lap_features.parquet")
    
    valid_compounds = {"C1", "C2", "C3", "C4", "C5", "C6"}
    clean = laps[
        (laps["IsClean"] == True) &
        (laps["ActualCompound"].isin(valid_compounds))
    ].copy()
    
    curves = {}
    group_cols = ["Season", "RoundNumber", "EventName", "Driver", "Team", "StintNumber"]
    
    for key, grp in clean.groupby(group_cols, dropna=False):
        grp = grp.sort_values("LapNumber")
        
        if len(grp) < min_laps:
            continue
        
        season, rnd, event, driver, team, stint_num = key
        compound = grp["ActualCompound"].iloc[0]
        
        # Use SmoothedLapTime if available, else FuelCorrectedLapTime
        if "SmoothedLapTime" in grp.columns and grp["SmoothedLapTime"].notna().any():
            times = grp["SmoothedLapTime"].values
        else:
            times = grp["FuelCorrectedLapTime"].values
        
        # Normalize: delta to first lap (so all curves start at 0)
        normalized = times - times[0]
        
        stint_id = f"{season}_{rnd}_{driver}_S{stint_num}"
        curves[stint_id] = {
            "curve": normalized,
            "season": season,
            "round": rnd,
            "event": event,
            "driver": driver,
            "team": team,
            "stint": stint_num,
            "compound": compound,
            "n_laps": len(normalized),
        }
    
    logger.info(f"  Extracted {len(curves)} stint curves (min {min_laps} laps)")
    return curves


def build_similarity_matrix(curves: dict, max_stints: int = 300) -> tuple:
    """
    Build pairwise DTW distance matrix for a sample of stints.
    (Full matrix for all stints would be O(n²) which is slow for 3000+)
    """
    logger.info(f"Building DTW similarity matrix (max {max_stints} stints)...")
    
    # Sample stints stratified by compound
    all_ids = list(curves.keys())
    
    if len(all_ids) > max_stints:
        # Stratified sample
        rng = np.random.default_rng(42)
        compounds = {}
        for sid, info in curves.items():
            c = info["compound"]
            compounds.setdefault(c, []).append(sid)
        
        per_compound = max_stints // len(compounds)
        sampled = []
        for comp, ids in compounds.items():
            n = min(per_compound, len(ids))
            sampled.extend(rng.choice(ids, n, replace=False).tolist())
        
        selected_ids = sampled[:max_stints]
    else:
        selected_ids = all_ids
    
    n = len(selected_ids)
    logger.info(f"  Computing {n*(n-1)//2} pairwise DTW distances...")
    
    # Compute pairwise distances
    dist_matrix = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2
    computed = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance_normalized(
                curves[selected_ids[i]]["curve"],
                curves[selected_ids[j]]["curve"],
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            computed += 1
            
            if computed % 5000 == 0:
                logger.info(f"    {computed}/{total_pairs} pairs computed...")
    
    logger.info(f"  ✓ Distance matrix: {n}×{n}")
    
    return dist_matrix, selected_ids


def find_similar_stints(
    curves: dict,
    query_id: str,
    all_ids: list = None,
    top_k: int = 5,
) -> list:
    """Find the k most similar stints to a query stint using DTW."""
    if all_ids is None:
        all_ids = [k for k in curves.keys() if k != query_id]
    
    query_curve = curves[query_id]["curve"]
    
    distances = []
    for sid in all_ids:
        if sid == query_id:
            continue
        d = dtw_distance_normalized(query_curve, curves[sid]["curve"])
        distances.append((sid, d))
    
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]


def run_dtw_analysis(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    features_dir = Path(config["paths"]["features"])
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  DYNAMIC TIME WARPING ANALYSIS")
    logger.info("=" * 60)
    
    # Extract curves
    curves = extract_stint_curves(features_dir, min_laps=8)
    
    # Build similarity matrix on a sample
    dist_matrix, selected_ids = build_similarity_matrix(curves, max_stints=200)
    
    # Cluster the stints based on DTW distances
    logger.info("\n  Clustering stints by degradation shape...")
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method="ward")
    
    # Find optimal k
    best_k, best_sil = 2, -1
    for k in range(2, 8):
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(dist_matrix, labels, metric="precomputed")
        logger.info(f"    k={k}: silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    labels = fcluster(linkage_matrix, best_k, criterion="maxclust")
    logger.info(f"  Optimal k={best_k} (silhouette={best_sil:.3f})")
    
    # Characterize clusters
    for cid in range(1, best_k + 1):
        cluster_ids = [selected_ids[i] for i in range(len(labels)) if labels[i] == cid]
        compounds = [curves[sid]["compound"] for sid in cluster_ids]
        avg_laps = np.mean([curves[sid]["n_laps"] for sid in cluster_ids])
        
        # Average final delta (how much slower at end vs start)
        final_deltas = [curves[sid]["curve"][-1] for sid in cluster_ids]
        avg_delta = np.mean(final_deltas)
        
        compound_dist = pd.Series(compounds).value_counts().to_dict()
        
        if avg_delta > 3:
            label = "High Degradation"
        elif avg_delta > 1:
            label = "Medium Degradation"
        else:
            label = "Low Degradation"
        
        logger.info(f"  Cluster {cid} — {label}: n={len(cluster_ids)}, "
                    f"avg_final_delta={avg_delta:.2f}s, avg_laps={avg_laps:.0f}, "
                    f"compounds={compound_dist}")
    
    # Example: find similar stints to a specific query
    logger.info("\n  Example stint similarity search:")
    example_ids = [sid for sid in selected_ids if "2024" in sid][:3]
    
    example_matches = {}
    for qid in example_ids:
        info = curves[qid]
        matches = find_similar_stints(curves, qid, selected_ids, top_k=3)
        
        logger.info(f"\n  Query: {qid} ({info['compound']}, {info['event']}, {info['n_laps']} laps)")
        match_list = []
        for mid, dist in matches:
            m = curves[mid]
            logger.info(f"    → {mid} ({m['compound']}, {m['event']}, "
                       f"{m['n_laps']} laps) DTW={dist:.4f}")
            match_list.append({
                "stint_id": mid,
                "compound": m["compound"],
                "event": m["event"],
                "driver": m["driver"],
                "dtw_distance": round(float(dist), 4),
            })
        
        example_matches[qid] = {
            "query_compound": info["compound"],
            "query_event": info["event"],
            "query_driver": info["driver"],
            "matches": match_list,
        }
    
    # Save dendrogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(linkage_matrix, no_labels=True, ax=ax,
                   color_threshold=linkage_matrix[-best_k+1, 2])
        ax.set_title(f"Stint Similarity Dendrogram (DTW, k={best_k})", fontsize=14)
        ax.set_ylabel("Ward Distance")
        plt.tight_layout()
        fig.savefig(output_dir / "dtw_dendrogram.png", dpi=150)
        plt.close()
        logger.info("\n  ✓ dtw_dendrogram.png")
    except ImportError:
        pass
    
    # Save results
    with open(output_dir / "dtw_example_matches.json", "w") as f:
        json.dump(example_matches, f, indent=2)
    logger.info(f"  ✓ dtw_example_matches.json")
    
    # Save distance matrix metadata
    dtw_meta = {
        "n_stints_analyzed": len(selected_ids),
        "optimal_k": best_k,
        "silhouette_score": round(float(best_sil), 3),
        "total_stints_available": len(curves),
    }
    with open(output_dir / "dtw_analysis_meta.json", "w") as f:
        json.dump(dtw_meta, f, indent=2)
    logger.info(f"  ✓ dtw_analysis_meta.json")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    run_dtw_analysis()
