"""
preprocessing.py - Functions for cleaning and preparing ProcessedVignette objects.
"""
import open3d as o3d
import numpy as np
from typing import Optional, Dict
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Import our custom data structure
from pipeline.vignette_data import ProcessedVignette

def preprocess_vignette(
    raw_vignette: ProcessedVignette,
    confidence_threshold: int = 1,
    voxel_size: float = 0.01,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    use_voxel_downsampling: bool = True,
    save_path: Optional[str] = None,
) -> Optional[ProcessedVignette]:
    """
    Runs a multi-step preprocessing pipeline on a ProcessedVignette.
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
    if use_voxel_downsampling:
        print(f"Downsampling with voxel size: {voxel_size}...")
        pcd_downsampled = pcd_pre_downsample.voxel_down_sample(voxel_size)
    else:
        print("Skipping vocel downsampling...")
        pcd_downsampled = pcd_pre_downsample
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

    if save_path is not None:
        processed_vignette.save(save_path)
        print(f"Saved vignette to {save_path}")
    else:
        print("Vignette not saved.")
    
    print("--- Preprocessing complete. Returning new clean vignette. ---")
    return processed_vignette


# Suggest the optimum cluster_eps for DBSCAN
# May use this as a starting value, and let use tweak it
def suggest_eps(
    vignette: ProcessedVignette, 
    k=8, 
    percentile=90
):
    points = vignette.points
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nbrs.kneighbors(points)
    # mean distance to k neighbors per point
    dk = dists[:, -1] # kth neighbor distance
    return np.percentile(dk, percentile)

# identify components with DBSCAN
def identify_components(
    vignette: ProcessedVignette,
    cluster_eps: float = 0.05,  # Maximum neighborhood radius to consider points neighbors
    min_cluster_points: int = 50, # Minimum neighbors 
    auto_save: bool = False
) -> int:
    """
    Identifies distinct geometric components in the vignette using DBSCAN clustering.

    This function adds a 'component_id' per-point attribute to the vignette,
    where -1 represents noise points.
    """
    print(f"Identifying components with eps={cluster_eps}...")
    
    clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_points).fit(vignette.points)
    labels = clustering.labels_
    
    # Store the result as a per-point attribute in the vignette
    vignette.set_attribute('component_id', labels, auto_save=auto_save)
    
    unique_labels = set(labels)
    num_components = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    print(f"   - Found {num_components} subcomponents and assigned 'component_id' attribute.")
    return num_components