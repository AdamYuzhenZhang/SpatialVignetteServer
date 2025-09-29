"""
preprocessing.py - Functions for cleaning and preparing ProcessedVignette objects.
"""
import open3d as o3d
import numpy as np
from typing import Optional, Dict

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

    This function is designed to be robust and future-proof. It automatically handles
    any and all per-point attributes present in the raw vignette.

    The pipeline consists of:
    1. (Optional) Filtering points below a confidence threshold, if confidence data exists.
    2. Downsampling the point cloud to make it uniform and efficient.
    3. Removing statistical outliers (effective against "flying pixels").
    4. Estimating normals, required for many abstraction algorithms.

    Args:
        raw_vignette: The input ProcessedVignette object containing raw point data.
        confidence_threshold: The minimum confidence value to keep (used if 'confidence' attribute exists).
        voxel_size: The size of the grid for downsampling (in meters).
        sor_nb_neighbors: The number of neighbors for statistical outlier removal.
        sor_std_ratio: The standard deviation ratio for outlier removal.

    Returns:
        A new, cleaned, and preprocessed ProcessedVignette object, or None if the
        input vignette is empty.
    """
    if raw_vignette.n_points == 0:
        print("Error: Input vignette is empty.")
        return None

    print(f"--- Starting Preprocessing ---")
    print(f"Initial points: {raw_vignette.n_points}")

    # Start with copies of the raw data arrays
    points = raw_vignette.points
    colors = raw_vignette.colors
    # Deep copy all custom attributes to avoid modifying the original vignette
    attributes = {name: values.copy() for name, values in raw_vignette.attributes.items()}

    # --- 1. (Optional) Confidence Filtering ---
    if 'confidence' in attributes:
        print(f"Filtering points with confidence < {confidence_threshold}...")
        keep_mask = attributes['confidence'] >= confidence_threshold
        
        # Apply the mask to core geometry and all custom attributes
        points = points[keep_mask]
        colors = colors[keep_mask]
        for name, values in attributes.items():
            attributes[name] = values[keep_mask]
        
        print(f"Points after confidence filtering: {len(points)}")
        if len(points) == 0:
            print("Warning: No points left after confidence filtering.")
            return None
    else:
        print("Info: No 'confidence' attribute found, skipping confidence filtering.")

    # Create an Open3D point cloud from the potentially filtered data
    pcd_pre_downsample = o3d.geometry.PointCloud()
    pcd_pre_downsample.points = o3d.utility.Vector3dVector(points)
    pcd_pre_downsample.colors = o3d.utility.Vector3dVector(colors)

    # --- 2. Voxel Downsampling ---
    print(f"Downsampling with voxel size: {voxel_size}...")
    pcd_downsampled = pcd_pre_downsample.voxel_down_sample(voxel_size)
    print(f"Points after downsampling: {len(pcd_downsampled.points)}")

    # --- 3. Statistical Outlier Removal (SOR) ---
    print("Removing statistical outliers...")
    pcd_cleaned, _ = pcd_downsampled.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio
    )
    print(f"Points after outlier removal: {len(pcd_cleaned.points)}")

    # --- 4. Normal Estimation ---
    print("Estimating normals...")
    pcd_cleaned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    # Orient normals consistently. This is important for meshing and lighting.
    pcd_cleaned.orient_normals_consistent_tangent_plane(100)
    print("Normals estimated and oriented.")

    # --- 5. Reconcile ALL Custom Attributes ---
    # Downsampling and outlier removal have changed the point count. We must now
    # create new attribute arrays that correspond to the final, cleaned points.
    # We do this by finding the nearest neighbor from the pre-downsampled cloud
    # for each point in the final cloud and inheriting its attributes.
    print("Reconciling custom attributes...")
    final_attributes: Dict[str, np.ndarray] = {}
    if attributes: # Only proceed if there are custom attributes to reconcile
        # Build a KDTree on the point cloud just before downsampling for efficient search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_pre_downsample)
        
        final_points_np = np.asarray(pcd_cleaned.points)
        
        # Find the indices of the nearest neighbors in the pre-downsampled cloud
        # This is much faster than looping point by point in Python
        indices = np.array([pcd_tree.search_knn_vector_3d(pt, 1)[1][0] for pt in final_points_np])

        # Use these indices to create the new attribute arrays
        for name, original_values in attributes.items():
            final_attributes[name] = original_values[indices]
        print(f"Reconciled {len(attributes)} custom attributes.")

    # --- 6. Create the new, clean ProcessedVignette object ---
    processed_vignette = ProcessedVignette(
        points=np.asarray(pcd_cleaned.points),
        colors=np.asarray(pcd_cleaned.colors),
        normals=np.asarray(pcd_cleaned.normals),
        # CRITICAL: Preserve the metadata from the original vignette
        metadata=raw_vignette.metadata,
        # Unpack the dictionary of final, reconciled attributes
        **final_attributes
    )
    
    print("--- Preprocessing complete. Returning new clean vignette. ---")
    return processed_vignette
