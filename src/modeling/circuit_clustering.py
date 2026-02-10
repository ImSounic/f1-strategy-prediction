"""
Circuit Similarity Clustering v2 (Agglomerative)
==================================================
FIXED: Uses median aggregation and filters outlier stints (|slope| > 1.0)
to prevent rain/anomaly stints from distorting circuit profiles.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_circuit_profiles(circuit_csv: Path, features_dir: Path) -> pd.DataFrame:
    """Build circuit profiles using MEDIAN aggregation and outlier filtering."""
    logger.info("Building circuit profiles (v2: robust aggregation)...")
    
    circuits = pd.read_csv(circuit_csv)
    stints = pd.read_parquet(features_dir / "stint_features.parquet")
    
    # Map EventName to circuit_key
    event_to_key = dict(zip(circuits['circuit_name'], circuits['circuit_key']))
    stints['circuit_key'] = stints['EventName'].map(event_to_key)
    
    # ── Filter outlier stints ──
    pre_filter = len(stints)
    stints = stints[
        (stints['StintLength'] >= 5) &           # min 5 laps for meaningful slope
        (stints['DegSlope'].abs() < 1.0) &        # remove extreme outliers
        (stints['Compound'].isin(['C1','C2','C3','C4','C5','C6']))
    ]
    logger.info(f"  Filtered stints: {pre_filter} → {len(stints)} "
                f"(removed {pre_filter - len(stints)} outliers/short stints)")
    
    # Circuit physical characteristics (stable across seasons)
    circuit_chars = circuits.groupby("circuit_key").agg(
        circuit_name=("circuit_name", "first"),
        asphalt_abrasiveness=("asphalt_abrasiveness", "mean"),
        asphalt_grip=("asphalt_grip", "mean"),
        traction_demand=("traction_demand", "mean"),
        braking_severity=("braking_severity", "mean"),
        lateral_forces=("lateral_forces", "mean"),
        tyre_stress=("tyre_stress", "mean"),
        downforce_level=("downforce_level", "mean"),
        track_evolution=("track_evolution", "mean"),
        circuit_length_km=("circuit_length_km", "mean"),
        pit_loss_seconds=("pit_loss_seconds", "mean"),
    ).reset_index()
    
    # Degradation behavior — MEDIAN to resist outliers
    deg_profile = stints.groupby("circuit_key").agg(
        median_deg_slope=("DegSlope", "median"),
        q25_deg_slope=("DegSlope", lambda x: x.quantile(0.25)),
        q75_deg_slope=("DegSlope", lambda x: x.quantile(0.75)),
        median_stint_length=("StintLength", "median"),
        median_deg_curvature=("DegCurvature", "median"),
        cliff_rate=("CliffDetected", "mean"),
        n_stints=("DegSlope", "count"),
    ).reset_index()
    
    # Incident rates
    incidents = pd.read_parquet(features_dir / "incident_features.parquet")
    race_circuit = circuits[["season", "round_number", "circuit_key"]].rename(
        columns={"season": "Season", "round_number": "RoundNumber"}
    )
    incidents = incidents.merge(race_circuit, on=["Season", "RoundNumber"], how="left")
    
    incident_profile = incidents.groupby("circuit_key").agg(
        sc_rate=("HasSC", "mean"),
        vsc_rate=("HasVSC", "mean"),
    ).reset_index()
    
    profiles = circuit_chars.merge(deg_profile, on="circuit_key", how="left")
    profiles = profiles.merge(incident_profile, on="circuit_key", how="left")
    profiles = profiles.fillna(0)
    
    # Verify no negative medians
    neg_median = (profiles['median_deg_slope'] < 0).sum()
    logger.info(f"  {len(profiles)} circuit profiles | "
                f"Negative median slopes: {neg_median} | "
                f"Median deg range: [{profiles['median_deg_slope'].min():.4f}, "
                f"{profiles['median_deg_slope'].max():.4f}]")
    
    return profiles


def find_optimal_clusters(X_scaled, linkage_matrix, max_k=8):
    """Find optimal number of clusters using silhouette score."""
    best_k, best_sil = 2, -1
    results = {}
    for k in range(2, min(max_k + 1, len(X_scaled))):
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        results[k] = {"silhouette": sil, "calinski_harabasz": ch}
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    logger.info(f"  Cluster search:")
    for k, scores in results.items():
        marker = " ←" if k == best_k else ""
        logger.info(f"    k={k}: silhouette={scores['silhouette']:.3f}, "
                    f"CH={scores['calinski_harabasz']:.1f}{marker}")
    
    return best_k, results


def cluster_circuits(profiles: pd.DataFrame):
    logger.info("Clustering circuits (v2: optimal k selection)...")
    
    feature_cols = [
        "asphalt_abrasiveness", "asphalt_grip", "traction_demand",
        "braking_severity", "lateral_forces", "tyre_stress",
        "downforce_level", "track_evolution", "circuit_length_km",
        "median_deg_slope", "q25_deg_slope", "q75_deg_slope",
        "median_stint_length", "median_deg_curvature", "cliff_rate", "sc_rate",
    ]
    
    available = [c for c in feature_cols if c in profiles.columns]
    X = profiles[available].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    linkage_matrix = linkage(X_scaled, method="ward")
    
    # Find optimal k
    best_k, cluster_search = find_optimal_clusters(X_scaled, linkage_matrix)
    logger.info(f"  Optimal k: {best_k}")
    
    cluster_labels = fcluster(linkage_matrix, best_k, criterion="maxclust")
    profiles["Cluster"] = cluster_labels
    
    sil_score = silhouette_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    
    logger.info(f"  Silhouette Score:  {sil_score:.3f}")
    logger.info(f"  Calinski-Harabasz: {ch_score:.1f}")
    
    # Print clusters with characterization
    for cid in sorted(profiles["Cluster"].unique()):
        members = profiles[profiles["Cluster"] == cid]
        names = members["circuit_name"].tolist()
        avg_deg = members["median_deg_slope"].mean()
        avg_stress = members["tyre_stress"].mean()
        avg_abr = members["asphalt_abrasiveness"].mean()
        
        # Auto-label cluster type
        if avg_deg > 0.08:
            label = "High Degradation"
        elif avg_deg > 0.04:
            label = "Medium Degradation"
        elif avg_deg > 0.02:
            label = "Low Degradation"
        else:
            label = "Minimal Degradation"
        
        if avg_stress >= 4:
            label += " / High Stress"
        
        logger.info(f"  Cluster {cid} — {label} "
                    f"(deg: {avg_deg:.4f} s/lap, abr: {avg_abr:.1f}, stress: {avg_stress:.1f}):")
        for name in names:
            logger.info(f"    - {name}")
    
    # Save dendrogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(16, 9))
        dendrogram(
            linkage_matrix,
            labels=profiles["circuit_key"].tolist(),
            leaf_rotation=45,
            leaf_font_size=9,
            color_threshold=linkage_matrix[-best_k + 1, 2] if best_k > 1 else 0,
            ax=ax,
        )
        ax.set_title(f"Circuit Similarity Dendrogram (k={best_k}, Silhouette={sil_score:.3f})",
                     fontsize=14)
        ax.set_ylabel("Ward Distance")
        plt.tight_layout()
        fig.savefig("models/circuit_dendrogram.png", dpi=150)
        plt.close()
        logger.info("  ✓ Dendrogram saved")
    except ImportError:
        logger.warning("  matplotlib not available")
    
    metrics = {
        "n_clusters": best_k,
        "silhouette_score": float(sil_score),
        "calinski_harabasz_score": float(ch_score),
        "n_circuits": len(profiles),
        "features_used": available,
        "cluster_search": {str(k): v for k, v in cluster_search.items()},
    }
    
    assignments = {}
    for _, row in profiles.iterrows():
        assignments[row["circuit_key"]] = {
            "cluster": int(row["Cluster"]),
            "circuit_name": row["circuit_name"],
            "median_deg_slope": float(row["median_deg_slope"]),
        }
    
    return profiles, linkage_matrix, metrics, assignments


def run_circuit_clustering(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    
    features_dir = Path(config["paths"]["features"])
    circuit_csv = Path(config["paths"]["raw"]["supplementary"]) / "pirelli_circuit_characteristics.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  CIRCUIT SIMILARITY CLUSTERING v2")
    logger.info("=" * 60)
    
    profiles = prepare_circuit_profiles(circuit_csv, features_dir)
    profiles, linkage_matrix, metrics, assignments = cluster_circuits(profiles)
    
    with open(model_dir / "circuit_clusters.json", "w") as f:
        json.dump(assignments, f, indent=2)
    logger.info(f"  ✓ Clusters saved")
    
    with open(model_dir / "circuit_cluster_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  ✓ Evaluation saved")
    
    profiles.to_parquet(model_dir / "circuit_profiles.parquet", index=False)
    logger.info(f"  ✓ Profiles saved")


if __name__ == "__main__":
    run_circuit_clustering()
