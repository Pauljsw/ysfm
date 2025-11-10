"""
3D Crack Clustering and Deduplication

Clusters crack points in 3D space to:
1. Remove duplicates (same crack detected in multiple images)
2. Merge fragments (one crack split across multiple detections)
3. Assign unique ID to each physical crack
4. Measure each crack in 3D

Uses DBSCAN clustering on dense point cloud crack points.

Usage:
    python -m src.cluster_cracks_3d \
        --crack-cloud outputs/dense_masked_cloud.ply \
        --output outputs/crack_clusters.json \
        --eps 50 \
        --min-samples 10
"""
import numpy as np
import json
import logging
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import csv

logger = logging.getLogger(__name__)


def load_crack_points_from_ply(ply_path: str, crack_color: Tuple[int, int, int] = (255, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load crack points from colored point cloud.

    Args:
        ply_path: Path to PLY with colored crack points
        crack_color: RGB color used for crack points

    Returns:
        (crack_xyz, all_xyz): Crack points and all points
    """
    logger.info(f"Loading point cloud: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Convert colors to uint8
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    logger.info(f"  Total points: {len(points):,}")

    # Extract crack points (matching crack_color)
    crack_mask = np.all(colors == crack_color, axis=1)
    crack_points = points[crack_mask]

    logger.info(f"  Crack points: {len(crack_points):,} ({len(crack_points)/len(points)*100:.1f}%)")

    return crack_points, points


def cluster_cracks_3d(
    crack_points: np.ndarray,
    eps: float = 50.0,
    min_samples: int = 10,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Cluster crack points in 3D using DBSCAN.

    Args:
        crack_points: Crack points (N, 3) in mm
        eps: Maximum distance between points in same cluster (mm)
        min_samples: Minimum points to form cluster
        metric: Distance metric

    Returns:
        labels: Cluster labels for each point (-1 = noise)
    """
    logger.info("Clustering crack points...")
    logger.info(f"  DBSCAN parameters: eps={eps}mm, min_samples={min_samples}")

    if len(crack_points) == 0:
        return np.array([])

    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = clustering.fit_predict(crack_points)

    # Statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    logger.info(f"  Clusters found: {n_clusters}")
    logger.info(f"  Noise points: {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")

    return labels


def measure_cluster(cluster_points: np.ndarray, cluster_id: int) -> Dict:
    """
    Measure a single crack cluster in 3D.

    Args:
        cluster_points: Points in cluster (N, 3)
        cluster_id: Cluster ID

    Returns:
        Measurement dictionary
    """
    n_points = len(cluster_points)

    # Bounding box
    bbox_min = cluster_points.min(axis=0)
    bbox_max = cluster_points.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Centroid
    centroid = cluster_points.mean(axis=0)

    # Length (maximum pairwise distance)
    if n_points >= 2:
        # For large clusters, sample points to speed up
        if n_points > 1000:
            sample_idx = np.random.choice(n_points, 1000, replace=False)
            sample_points = cluster_points[sample_idx]
        else:
            sample_points = cluster_points

        pairwise_dist = pdist(sample_points)
        max_length = np.max(pairwise_dist)
    else:
        max_length = 0.0

    # Volume (convex hull if enough points)
    volume = 0.0
    if n_points >= 4:
        try:
            hull = ConvexHull(cluster_points)
            volume = hull.volume
        except Exception as e:
            logger.debug(f"ConvexHull failed for cluster {cluster_id}: {e}")

    # Average width (perpendicular to main axis)
    # Simple estimate: use std of distances from centroid
    distances_from_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
    avg_width = np.mean(distances_from_centroid) * 2  # Diameter

    return {
        'cluster_id': int(cluster_id),
        'n_points': int(n_points),
        'length_mm': float(max_length),
        'avg_width_mm': float(avg_width),
        'volume_mm3': float(volume),
        'centroid_x': float(centroid[0]),
        'centroid_y': float(centroid[1]),
        'centroid_z': float(centroid[2]),
        'bbox_min_x': float(bbox_min[0]),
        'bbox_min_y': float(bbox_min[1]),
        'bbox_min_z': float(bbox_min[2]),
        'bbox_max_x': float(bbox_max[0]),
        'bbox_max_y': float(bbox_max[1]),
        'bbox_max_z': float(bbox_max[2]),
        'bbox_size_x': float(bbox_size[0]),
        'bbox_size_y': float(bbox_size[1]),
        'bbox_size_z': float(bbox_size[2])
    }


def save_cluster_point_cloud(
    crack_points: np.ndarray,
    labels: np.ndarray,
    output_ply: str
):
    """
    Save clustered point cloud with different color per cluster.

    Args:
        crack_points: Crack points (N, 3)
        labels: Cluster labels (N,)
        output_ply: Output PLY path
    """
    logger.info(f"Saving clustered point cloud: {output_ply}")

    # Generate colors for each cluster
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Color map
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(n_clusters + 1, 3))

    # Assign colors
    point_colors = np.zeros((len(crack_points), 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label == -1:
            # Noise = gray
            point_colors[i] = [100, 100, 100]
        else:
            point_colors[i] = colors[label % len(colors)]

    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(crack_points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)

    o3d.io.write_point_cloud(output_ply, pcd, write_ascii=False)
    logger.info(f"  Saved {len(crack_points):,} points with {n_clusters} clusters")


def run_clustering(
    crack_cloud_ply: str,
    output_json: str,
    output_csv: str = None,
    output_clustered_ply: str = None,
    eps: float = 50.0,
    min_samples: int = 10,
    min_cluster_size: int = 50,
    crack_color: Tuple[int, int, int] = (255, 0, 0)
):
    """
    Run 3D crack clustering pipeline.

    Args:
        crack_cloud_ply: Input PLY with colored crack points
        output_json: Output JSON path
        output_csv: Output CSV path (optional)
        output_clustered_ply: Output clustered PLY (optional)
        eps: DBSCAN epsilon (mm)
        min_samples: DBSCAN min samples
        min_cluster_size: Minimum points to keep cluster
        crack_color: RGB color of crack points
    """
    logger.info("=" * 80)
    logger.info("3D Crack Clustering and Deduplication")
    logger.info("=" * 80)

    # Load crack points
    crack_points, all_points = load_crack_points_from_ply(crack_cloud_ply, crack_color)

    if len(crack_points) == 0:
        logger.warning("No crack points found! Check crack_color parameter.")
        return []

    # Cluster
    labels = cluster_cracks_3d(crack_points, eps, min_samples)

    # Measure each cluster
    measurements = []
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    logger.info("Measuring clusters...")

    for label in sorted(unique_labels):
        if label == -1:
            continue  # Skip noise

        cluster_mask = labels == label
        cluster_points = crack_points[cluster_mask]

        # Filter small clusters
        if len(cluster_points) < min_cluster_size:
            logger.debug(f"  Cluster {label}: {len(cluster_points)} points (too small, skipped)")
            continue

        measurement = measure_cluster(cluster_points, label)
        measurements.append(measurement)

        logger.info(f"  Cluster {label}: {measurement['n_points']} points, "
                   f"length={measurement['length_mm']:.1f}mm, "
                   f"width={measurement['avg_width_mm']:.1f}mm")

    logger.info(f"Total valid clusters: {len(measurements)}")

    # Save JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'metadata': {
            'total_crack_points': int(len(crack_points)),
            'total_clusters': int(n_clusters),
            'valid_clusters': len(measurements),
            'eps': eps,
            'min_samples': min_samples,
            'min_cluster_size': min_cluster_size
        },
        'clusters': measurements
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved JSON: {output_json}")

    # Save CSV
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', newline='') as csvfile:
            if measurements:
                fieldnames = measurements[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(measurements)

        logger.info(f"Saved CSV: {output_csv}")

    # Save clustered point cloud
    if output_clustered_ply:
        ply_path = Path(output_clustered_ply)
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        save_cluster_point_cloud(crack_points, labels, str(ply_path))

    logger.info("=" * 80)
    logger.info("Clustering complete!")
    logger.info("=" * 80)

    return measurements


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='3D crack clustering')
    parser.add_argument('--crack-cloud', required=True,
                       help='Input PLY with colored crack points')
    parser.add_argument('--output', required=True,
                       help='Output JSON path')
    parser.add_argument('--output-csv', default=None,
                       help='Output CSV path (optional)')
    parser.add_argument('--output-clustered-ply', default=None,
                       help='Output clustered PLY (optional)')
    parser.add_argument('--eps', type=float, default=50.0,
                       help='DBSCAN epsilon in mm (default: 50)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='DBSCAN min samples (default: 10)')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                       help='Minimum points to keep cluster (default: 50)')
    parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
                       help='RGB color of crack points (default: 255 0 0)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        measurements = run_clustering(
            args.crack_cloud,
            args.output,
            args.output_csv,
            args.output_clustered_ply,
            args.eps,
            args.min_samples,
            args.min_cluster_size,
            tuple(args.crack_color)
        )

        print(f"\n✅ Clustering complete!")
        print(f"   Valid clusters: {len(measurements)}")
        if measurements:
            total_length = sum(m['length_mm'] for m in measurements)
            print(f"   Total crack length: {total_length:.1f} mm")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
