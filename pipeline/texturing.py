"""
texturing.py - Functions for creating textured meshes from abstracted primitives.

This module takes a vignette that has already been analyzed by the abstraction
pipeline and projects the original RGB image onto the surfaces of the detected
primitives, creating a set of textured 3D "scraps" similar to Mental Canvas.
"""

import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Optional, Dict, Any
from scipy.spatial import ConvexHull

# Import our custom data structure
from .vignette_data import ProcessedVignette


def project_points_to_2d(points_3d, intrinsics):
    """
    Projects 3D points (in original camera space) to 2D image coordinates.
    
    MATH: This function implements the standard camera projection matrix formula.
    The intrinsic matrix (K) maps 3D camera coordinates [X, Y, Z] to
    homogeneous 2D image coordinates [u*d, v*d, d]. By dividing by the third
    component (d, which is the depth), we get the final pixel coordinates [u, v].
    
    Args:
        points_3d: An (N, 3) NumPy array of 3D points.
        intrinsics: The 3x3 camera intrinsic matrix.

    Returns:
        An (N, 2) NumPy array of 2D pixel coordinates.
    """
    print(f"      - Projecting {len(points_3d)} points using intrinsics:\n{intrinsics.round(2)}")
    # Project points onto the image plane
    projected = intrinsics @ points_3d.T
    
    # Get the depth value (d) for each point
    d = projected[2, :] + 1e-6 # Add epsilon to avoid division by zero
    
    # Normalize to get final pixel coordinates (u, v)
    u = projected[0, :] / d
    v = projected[1, :] / d
    
    result_2d = np.vstack((u, v)).T
    print(f"      - Raw 2D pixel coordinates (first corner):\n{result_2d[0].round(2)}")
    return result_2d


def create_textured_planes(
    vignette: ProcessedVignette,
    rgb_image_path: str,
    output_dir: str # Note: output_dir is kept for API consistency but not used in this version.
) -> None:
    """
    Generates textured meshes for all dominant planes found in a vignette.

    This is a simplified implementation for debugging. It textures each plane
    mesh using the full, original RGB image. It does NOT crop or create masks.
    """
    print("--- Starting SIMPLIFIED Textured Plane Creation (Full Image Method) ---")
    vignette.clear_abstractions('textured_planes', auto_save=False)
    
    # === Step 1: Load Prerequisites ===
    print("\n[Step 1/5] Loading prerequisites...")
    plane_abstractions = vignette.get_abstractions('planes')
    if not plane_abstractions:
        print("   - No planes found in vignette to texture. Aborting.")
        return

    try:
        original_image = Image.open(rgb_image_path)
        original_image_size = original_image.size
        print(f"   - Loaded original RGB image: {rgb_image_path} (size: {original_image_size})")
    except FileNotFoundError:
        print(f"   - ERROR: Could not find original RGB image at: {rgb_image_path}")
        return

    capture_meta = vignette.metadata.get('capture_metadata', {})
    intrinsics = np.array(capture_meta.get('camera_intrinsics', {}).get('columns', np.identity(3))).T
    center_offset = np.array(capture_meta.get('center_offset', [0,0,0]))

    if np.array_equal(intrinsics, np.identity(3)) or np.allclose(center_offset, [0,0,0]):
         print("   - ERROR: Camera intrinsics or center_offset not found in vignette metadata.")
         return
    print(f"   - Successfully loaded camera intrinsics and center_offset: {center_offset.round(2)}")
    
    textured_planes_list = []

    for plane_info in plane_abstractions:
        plane_id = plane_info['plane_id']
        print(f"\n--- Processing Plane ID: {plane_id} ---")
        if "obb_center" not in plane_info:
            print("   - Skipping plane, missing OBB data.")
            continue

        # === Step 2: Define 3D Mesh Vertices ===
        print("[Step 2/5] Defining 3D mesh vertices...")
        center, rotation, extent = np.array(plane_info['obb_center']), np.array(plane_info['obb_rotation']), np.array(plane_info['obb_extent'])
        axes = [rotation[:, 0], rotation[:, 1], rotation[:, 2]]
        sorted_indices = np.argsort(extent)
        major_axis, minor_axis = axes[sorted_indices[2]], axes[sorted_indices[1]]
        major_half, minor_half = extent[sorted_indices[2]] / 2.0, extent[sorted_indices[1]] / 2.0
        v1 = center + major_axis*major_half + minor_axis*minor_half
        v2 = center - major_axis*major_half + minor_axis*minor_half
        v3 = center - major_axis*major_half - minor_axis*minor_half
        v4 = center + major_axis*major_half - minor_axis*minor_half
        corners_3d_centered = np.array([v1, v2, v3, v4])
        print(f"   - Calculated 4 centered 3D corner vertices (first corner):\n{corners_3d_centered[0].round(3)}")

        # === Step 3: Calculate UV Coordinates ===
        print("[Step 3/5] Calculating UV coordinates...")
        # MATH: To get the correct UVs, we must project the UN-CENTERED 3D corners.
        # We add the center_offset back to the vertices to place them in their
        # original position relative to the camera before projection.
        corners_3d_original = corners_3d_centered + center_offset
        print(f"   - Un-centered 3D corners by adding offset (first corner):\n{corners_3d_original[0].round(3)}")
        
        # Project the original-space 3D points to get 2D pixel coordinates.
        corners_2d = project_points_to_2d(corners_3d_original, intrinsics)
        
        # MATH: UV coordinates are normalized, where (0,0) is the top-left of the
        # texture and (1,1) is the bottom-right. We divide the 2D pixel coordinates
        # by the full image dimensions to get these normalized values.
        uv_corners = corners_2d / np.array(original_image_size)
        print(f"   - Normalized 2D pixel coordinates to get final UVs (first corner):\n{uv_corners[0].round(3)}")
        
        # === Step 4: Store Final Data in Vignette ===
        print("[Step 4/5] Storing textured plane data in vignette metadata...")
        textured_plane_data = {
            "plane_id": plane_id,
            "mesh_vertices": [v.tolist() for v in [v1, v2, v3, v4]],
            "mesh_faces": [[0, 1, 2], [0, 2, 3]],
            "mesh_uvs": uv_corners.tolist(),
            # This now points to the original RGB image, not a new asset.
            "texture_path": Path(rgb_image_path).name 
        }
        textured_planes_list.append(textured_plane_data)

    # === Step 5: Save Results ===
    print("\n[Step 5/5] Saving results to vignette metadata...")
    vignette.metadata['textured_abstractions'] = {'planes': textured_planes_list}
    vignette.save()
    print("\n--- Finished SIMPLIFIED Textured Plane Creation ---")

