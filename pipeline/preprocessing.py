"""
preprocessing.py - Functions for cleaning and preparing ProcessedVignette objects.

This module provides a pipeline to filter noise, remove outliers, and prepare
the point cloud for abstraction algorithms. It follows a functional approach,
taking a raw vignette as input and returning a new, cleaned vignette as output.
"""
import open3d as o3d
import numpy as np
from typing import Optional

# Import our custom data structure
from pipeline.vignette_data import ProcessedVignette

def preprocess_vignette(
    raw_vignette: ProcessedVignette,
    confidence_threshold: int = 1,
    voxel_size: float = 0.01,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0
) -> Optional[ProcessedVignette]:
    """
    Runs a multi-step preprocessing pipeline on a ProcessedVignette.

    The pipeline consists of:
    1. Filtering out points below a certain confidence level.
    2. Downsampling the point cloud to make it uniform and efficient.
    3. Removing statistical outliers (effective against "flying pixels").
    4. Estimating normals, which are required for many abstraction algorithms.

    Args:
        raw_vignette: The input ProcessedVignette object containing raw point data.
        confidence_threshold: The minimum confidence value to keep.
        voxel_size: The size of the grid for downsampling (in meters).
        sor_nb_neighbors: The number of neighbors to analyze for outlier removal.
        sor_std_ratio: The standard deviation ratio for outlier removal.

    Returns:
        A new, cleaned, and preprocessed ProcessedVignette object, or None if an
        error occurred.
    """
    if raw_vignette.points is None or len(raw_vignette.points) == 0:
        print("Error: Input vignette is empty.")
        return None
    if raw_vignette.confidence is None:
        print("Error: Input vignette is missing confidence data for filtering.")
        return None
    if len(raw_vignette.points) != len(raw_vignette.confidence):
        print("Error: Point cloud and confidence values mismatch.")
        return None

    # --- 1. Confidence Filtering ---
    # We create a boolean mask of which points to keep.
    print(f"Initial points: {len(raw_vignette.points)}")
    keep_mask = raw_vignette.confidence >= confidence_threshold
    
    # Apply the mask to all per-point attributes
    points_conf_filtered = raw_vignette.points[keep_mask]
    colors_conf_filtered = raw_vignette.colors[keep_mask]
    
    # Create a temporary Open3D point cloud for the next steps
    pcd_for_processing = o3d.geometry.PointCloud()
    pcd_for_processing.points = o3d.utility.Vector3dVector(points_conf_filtered)
    pcd_for_processing.colors = o3d.utility.Vector3dVector(colors_conf_filtered)
    
    print(f"Points after confidence filtering: {len(pcd_for_processing.points)}")
    if len(pcd_for_processing.points) == 0:
        print("Warning: No points left after confidence filtering.")
        return None

    # --- 2. Voxel Downsampling ---
    pcd_downsampled = pcd_for_processing.voxel_down_sample(voxel_size)
    print(f"Points after downsampling: {len(pcd_downsampled.points)}")

    # --- 3. Statistical Outlier Removal (SOR) ---
    pcd_cleaned, _ = pcd_downsampled.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio
    )
    print(f"Points after outlier removal: {len(pcd_cleaned.points)}")

    # --- 4. Normal Estimation ---
    pcd_cleaned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_cleaned.orient_normals_consistent_tangent_plane(100)
    print("Normals estimated.")

    # --- 5. Reconcile Custom Attributes ---
    # The number of points has changed. We need to find the confidence values
    # for the points that survived the cleaning process.
    # We use a KDTree for an efficient nearest neighbor search.
    print("Reconciling custom attributes...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_for_processing)
    
    cleaned_confidences = []
    # For each point in our final clean cloud, find its original index
    for point in pcd_cleaned.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        # Get the confidence value from the original confidence-filtered array
        cleaned_confidences.append(raw_vignette.confidence[keep_mask][idx[0]])

    # --- 6. Create the new, clean ProcessedVignette object ---
    processed_vignette = ProcessedVignette(
        points=np.asarray(pcd_cleaned.points),
        colors=np.asarray(pcd_cleaned.colors),
        normals=np.asarray(pcd_cleaned.normals),
        confidence=np.array(cleaned_confidences)
    )
    
    print("Preprocessing complete. Returning new clean vignette.")
    return processed_vignette