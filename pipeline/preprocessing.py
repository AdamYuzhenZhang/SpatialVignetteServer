"""
preprocessing.py - Functions for cleaning and preparing point clouds.

This module provides a pipeline to filter noise, remove outliers, and prepare
the point cloud for abstraction algorithms.
"""
import open3d as o3d
import numpy as np
from typing import Optional

def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    confidence_values: np.ndarray,
    confidence_threshold: int = 1,
    voxel_size: float = 0.005,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0
) -> Optional[o3d.geometry.PointCloud]:
    """
    Runs a multi-step preprocessing pipeline on a point cloud.

    The pipeline consists of:
    1. Filtering out points below a certain confidence level.
    2. Downsampling the point cloud to make it uniform and efficient.
    3. Removing statistical outliers (effective against "flying pixels").
    4. Estimating normals, which are required for many abstraction algorithms.

    Args:
        pcd: The raw input Open3D PointCloud.
        confidence_values: A NumPy array of confidence values (0, 1, or 2)
                           corresponding to each point in the pcd.
        confidence_threshold: The minimum confidence value to keep (e.g., 1 means
                              0 will be removed).
        voxel_size: The size of the grid for downsampling.
        sor_nb_neighbors: The number of neighbors to analyze for outlier removal.
        sor_std_ratio: The standard deviation ratio for outlier removal. Higher
                       values are less aggressive.

    Returns:
        The cleaned and preprocessed Open3D PointCloud, or None if the
        input is invalid.
    """
    if len(pcd.points) == 0 or len(pcd.points) != len(confidence_values):
        print("Error: Point cloud is empty or confidence values mismatch.")
        return None

    # --- 1. Confidence Filtering ---
    # Keep only the points with confidence >= threshold.
    print(f"Initial points: {len(pcd.points)}")
    indices_to_keep = np.where(confidence_values >= confidence_threshold)[0]
    pcd_conf_filtered = pcd.select_by_index(indices_to_keep)
    print(f"Points after confidence filtering: {len(pcd_conf_filtered.points)}")

    # --- 2. Voxel Downsampling ---
    # This makes the point cloud uniform and more efficient to process.
    pcd_downsampled = pcd_conf_filtered.voxel_down_sample(voxel_size)
    print(f"Points after downsampling: {len(pcd_downsampled.points)}")

    # --- 3. Statistical Outlier Removal (SOR) ---
    # This is the key step for removing "flying pixels" and noise.
    print("Removing statistical outliers...")
    pcd_cleaned, ind = pcd_downsampled.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio
    )
    print(f"Points after outlier removal: {len(pcd_cleaned.points)}")

    # --- 4. Normal Estimation ---
    # Normals describe the orientation of the surface at each point.
    # They are essential for many mesh and plane fitting algorithms.
    print("Estimating normals...")
    pcd_cleaned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_cleaned.orient_normals_consistent_tangent_plane(100)
    
    print("Preprocessing complete.")
    return pcd_cleaned
