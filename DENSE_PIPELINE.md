# Dense Point Cloud Pipeline with 3D Crack Clustering

Complete pipeline for 3D crack detection, deduplication, and measurement using dense COLMAP reconstruction.

## Overview

This pipeline addresses two key challenges:
1. **Duplication**: Same crack detected in multiple images
2. **Fragmentation**: Single crack split across multiple detections

**Solution**: 3D clustering in point cloud space to identify unique physical cracks.

---

## Pipeline Phases

### Phase 0: SFM (Sparse + Dense Reconstruction)

Generate camera poses and dense 3D reconstruction.

```bash
python -m src.pipeline sfm --config configs/simple.yaml
```

**Outputs:**
- `data/sfm/sparse/0/` - Camera poses and sparse point cloud
- `data/sfm/dense/fused.ply` or `fused_photometric.ply` - Dense point cloud

**Requirements:**
- RGB images in `data/rgb/`
- COLMAP installed

---

### Phase 1: YOLO Inference

Detect cracks in 2D images.

```bash
python -m src.pipeline infer --config configs/simple.yaml
```

**Outputs:**
- `data/yolo_masks/*.json` - Crack masks for each image

**Requirements:**
- YOLO weights (YOLOv8+ segmentation model)
- RGB images

---

### Phase 2: Pixel-to-MM Calibration

Calculate pixel scale for each image using depth information.

```bash
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json
```

**Outputs:**
- `calibration/pixel_scales.json` - Pixel-to-millimeter scales per image

**Requirements:**
- Aligned RGB-Depth pairs
- Camera calibration JSON

**Note:** This phase is independent of sparse/dense point clouds.

---

### Phase 3: Dense Point Cloud Mask Overlay

Project YOLO masks onto dense point cloud in 3D.

```bash
python -m src.point_cloud_overlay_dense \
  --dense-ply data/sfm/dense/fused.ply \
  --sparse-dir data/sfm/sparse/0 \
  --masks-dir data/yolo_masks \
  --output outputs/dense_masked_cloud.ply \
  --min-votes 1
```

**Parameters:**
- `--dense-ply`: Dense point cloud (fused.ply or fused_photometric.ply)
- `--sparse-dir`: Camera poses from sparse reconstruction
- `--masks-dir`: YOLO mask JSONs
- `--min-votes`: Minimum views where point must appear as crack (default: 1)

**Outputs:**
- `outputs/dense_masked_cloud.ply` - Point cloud with crack points colored red

**How it works:**
1. Load dense point cloud (millions of points)
2. For each 3D point:
   - Project to all camera views
   - Check if pixel is inside crack mask
   - Count votes across views
3. Color points red if votes ≥ min-votes

**Camera Models Supported:**
- SIMPLE_PINHOLE, PINHOLE
- SIMPLE_RADIAL, RADIAL
- OPENCV (with radial + tangential distortion)

---

### Phase 4: 3D Crack Clustering (NEW!)

Cluster crack points in 3D to remove duplicates and merge fragments.

```bash
python -m src.cluster_cracks_3d \
  --crack-cloud outputs/dense_masked_cloud.ply \
  --output outputs/crack_clusters.json \
  --output-csv outputs/crack_clusters.csv \
  --output-clustered-ply outputs/clustered_cracks.ply \
  --eps 50 \
  --min-samples 10 \
  --min-cluster-size 50
```

**Parameters:**
- `--crack-cloud`: Input PLY with colored crack points (from Phase 3)
- `--eps`: DBSCAN epsilon - maximum distance (mm) between points in same cluster
  - Larger = merge more aggressively
  - Smaller = more conservative
  - Typical: 20-100mm depending on crack spacing
- `--min-samples`: Minimum points to form dense region (DBSCAN core points)
- `--min-cluster-size`: Minimum points to keep cluster (filter tiny clusters)
- `--crack-color`: RGB color of crack points (default: 255 0 0 for red)

**Outputs:**
- `outputs/crack_clusters.json` - Cluster measurements (JSON)
- `outputs/crack_clusters.csv` - Cluster measurements (CSV)
- `outputs/clustered_cracks.ply` - Point cloud with different color per cluster (optional)

**Measurements per cluster:**
- `cluster_id`: Unique crack ID
- `n_points`: Number of 3D points in crack
- `length_mm`: Maximum pairwise distance (crack length)
- `avg_width_mm`: Average width estimate
- `volume_mm3`: Convex hull volume
- `centroid_x/y/z`: 3D centroid position
- `bbox_min/max/size_x/y/z`: Bounding box

**How it works:**
1. Extract red (crack) points from PLY
2. Run DBSCAN clustering in 3D space
3. Each cluster = one unique physical crack
4. Measure geometric properties of each cluster

**Benefits:**
- ✅ Same crack seen in multiple images → **1 cluster**
- ✅ Crack split across detections → **merged into 1 cluster**
- ✅ Noise points filtered out
- ✅ Unique ID per physical crack

---

### Phase 5 (Optional): 2D Measurement per Image

Measure cracks in individual 2D images (old method, for comparison).

```bash
python -m src.measure_cracks_simple \
  --masks-dir data/yolo_masks \
  --pixel-scales calibration/pixel_scales.json \
  --output outputs/measurements_2d.csv
```

**Outputs:**
- `outputs/measurements_2d.csv` - Per-image measurements (may have duplicates)

**Note:** This is the **old 2D method** without deduplication. Use Phase 4 for accurate results.

---

## Complete Workflow Example

```bash
# 0. SFM reconstruction (sparse + dense)
python -m src.pipeline sfm --config configs/simple.yaml

# 1. YOLO crack detection
python -m src.pipeline infer --config configs/simple.yaml

# 2. Pixel calibration (if using depth)
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json

# 3. Dense point cloud overlay
python -m src.point_cloud_overlay_dense \
  --dense-ply data/sfm/dense/fused.ply \
  --sparse-dir data/sfm/sparse/0 \
  --masks-dir data/yolo_masks \
  --output outputs/dense_masked_cloud.ply \
  --min-votes 1

# 4. 3D clustering (deduplication + measurement)
python -m src.cluster_cracks_3d \
  --crack-cloud outputs/dense_masked_cloud.ply \
  --output outputs/crack_clusters.json \
  --output-csv outputs/crack_clusters.csv \
  --output-clustered-ply outputs/clustered_cracks.ply \
  --eps 50 \
  --min-samples 10
```

---

## Parameter Tuning Guide

### DBSCAN Parameters

**`--eps` (epsilon):**
- Physical meaning: Maximum distance (mm) for two points to be neighbors
- **Too small**: Cracks split into multiple clusters
- **Too large**: Multiple cracks merged into one
- Start with: 50mm
- Adjust based on:
  - Crack spacing: Closer cracks need smaller eps
  - Point density: Denser clouds can use smaller eps

**`--min-samples`:**
- Physical meaning: Minimum neighbors for a "core" point
- **Too small**: More noise classified as cracks
- **Too large**: Small cracks ignored
- Start with: 10
- Adjust based on point density

**`--min-cluster-size`:**
- Post-processing filter: Remove tiny clusters
- Start with: 50
- Increase to filter more aggressively

### Testing Strategy

1. **Start conservative**: `--eps 30 --min-samples 20`
2. **Visualize**: Open `outputs/clustered_cracks.ply` in CloudCompare/MeshLab
3. **Check for issues**:
   - Same crack split? → Increase `--eps`
   - Different cracks merged? → Decrease `--eps`
   - Too much noise? → Increase `--min-samples` or `--min-cluster-size`

---

## Output Files Summary

| File | Phase | Description |
|------|-------|-------------|
| `data/sfm/sparse/0/` | 0 | Camera poses |
| `data/sfm/dense/fused.ply` | 0 | Dense point cloud |
| `data/yolo_masks/*.json` | 1 | 2D crack masks |
| `calibration/pixel_scales.json` | 2 | Pixel-to-mm scales |
| `outputs/dense_masked_cloud.ply` | 3 | Crack-colored point cloud |
| `outputs/crack_clusters.json` | 4 | **Final measurements (3D)** |
| `outputs/crack_clusters.csv` | 4 | **Final measurements (CSV)** |
| `outputs/clustered_cracks.ply` | 4 | Visualization (colored clusters) |

---

## Comparison: Sparse vs Dense

| Aspect | Sparse | Dense (New) |
|--------|--------|-------------|
| Point count | ~10K-100K | 1M-10M+ |
| Coverage | Feature points only | Full surface |
| Overlay method | Track-based | Projection-based |
| Suitable for | Quick preview | Production use |

**Recommendation**: Use **Dense** for final results.

---

## Troubleshooting

### "No crack points found"
- Check `--crack-color` matches the color used in Phase 3
- Verify Phase 3 output: open PLY and confirm red points exist

### "Too many clusters"
- Increase `--eps` to merge more aggressively
- Increase `--min-cluster-size` to filter small clusters

### "Cracks merged incorrectly"
- Decrease `--eps` to be more conservative
- Check point cloud quality (dense reconstruction may have noise)

### "Warning: Unsupported camera model"
- Update `point_cloud_overlay_dense.py` to support your camera model
- Or use fallback (may be less accurate)

### "Very slow"
- Use `--max-points` to test on subset first
- Dense clouds with millions of points take time (10-30 min typical)

---

## Visualization

### CloudCompare (Recommended)
```bash
# Install CloudCompare
sudo snap install cloudcompare

# Open files
cloudcompare outputs/dense_masked_cloud.ply
cloudcompare outputs/clustered_cracks.ply
```

### MeshLab
```bash
# Install MeshLab
sudo apt install meshlab

# Open files
meshlab outputs/clustered_cracks.ply
```

### Python (Open3D)
```python
import open3d as o3d

# Load and visualize
pcd = o3d.io.read_point_cloud("outputs/clustered_cracks.ply")
o3d.visualization.draw_geometries([pcd])
```

---

## Next Steps

1. **Validate results**: Compare cluster measurements with ground truth
2. **Export for GIS**: Convert centroids to GeoJSON for mapping
3. **Temporal analysis**: Run pipeline on multiple dates to track crack growth
4. **Automated reporting**: Generate markdown/PDF reports with measurements

---

## References

- COLMAP: https://colmap.github.io/
- DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
- YOLOv8: https://docs.ultralytics.com/

---

## Support

For issues or questions:
- Check existing issues: https://github.com/Pauljsw/ysfm/issues
- Create new issue with logs and parameters used
