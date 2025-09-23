# abstraction.py
# Core logic for the abstraction pipeline

from typing import List, Tuple
import numpy as np
import open3d as o3d


# --- 1. Structural Analysis: Finding the "Bones" ---

def extract_dominant_axis(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the dominant axis and dimensions of a point cloud using PCA.

    This is useful for understanding the primary orientation and scale of an object.
    It computes an Oriented Bounding Box (OBB) and returns its center, the
    primary axis vector (the longest side), and the extent (length, width, height).

    Args:
        pcd: The input Open3D PointCloud.

    Returns:
        A tuple containing:
        - center (np.ndarray): The center point of the bounding box.
        - primary_axis (np.ndarray): A vector representing the longest axis.
        - extent (np.ndarray): The dimensions [length, width, height] of the box.
    """
    # get_oriented_bounding_box() uses PCA internally to find the best fit.
    obb = pcd.get_oriented_bounding_box()
    
    # The rotation matrix contains the axes as its columns.
    # The extent tells us the length of the box along each of those axes.
    rotation = obb.R
    extent = obb.extent
    
    # Find the index of the longest side of the box
    longest_axis_index = np.argmax(extent)
    
    # The primary axis is the column in the rotation matrix corresponding to the longest side.
    primary_axis = rotation[:, longest_axis_index]
    
    print(f"Dominant Axis Analysis: Center={obb.center}, Axis={primary_axis}, Extent={extent}")
    return obb.center, primary_axis, extent


# --- 2. Planar Abstraction: Finding the "Surfaces" ---

def extract_dominant_planes(
    pcd: o3d.geometry.PointCloud,
    min_points_for_plane: int = 100,
    distance_threshold: float = 0.01
) -> List[Tuple[np.ndarray, o3d.geometry.PointCloud]]:
    """
    Finds multiple dominant planes in a point cloud using RANSAC.

    This method iteratively finds the largest plane, removes its points, and then
    searches for the next largest plane in the remaining points. It's a core
    technique for architectural analysis.

    Args:
        pcd: The input Open3D PointCloud.
        min_points_for_plane: The minimum number of points a plane needs to be considered valid.
        distance_threshold: The max distance a point can be from a plane to be an inlier.

    Returns:
        A list of tuples, where each tuple contains:
        - plane_equation (np.ndarray): The [a, b, c, d] coefficients of the plane.
        - plane_pcd (o3d.geometry.PointCloud): A point cloud of only the inlier points.
    """
    planes = []
    remaining_pcd = pcd
    
    while len(remaining_pcd.points) > min_points_for_plane:
        # segment_plane returns the equation and the indices of the inlier points
        plane_model, inlier_indices = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inlier_indices) < min_points_for_plane:
            # Not enough points left to form a significant plane
            break
            
        # Create a new point cloud containing only the plane's points
        plane_pcd = remaining_pcd.select_by_index(inlier_indices)
        planes.append((plane_model, plane_pcd))
        
        # Create a new point cloud with the plane points removed
        remaining_pcd = remaining_pcd.select_by_index(inlier_indices, invert=True)
        
        print(f"Found plane with {len(inlier_indices)} points. {len(remaining_pcd.points)} points remaining.")
        
    return planes


# --- 3. Volumetric Abstraction: Finding the "Mass" ---

def create_abstract_mesh(pcd: o3d.geometry.PointCloud, alpha: float = 0.05) -> o3d.geometry.TriangleMesh:
    """
    Creates a simplified, "watertight" mesh from a point cloud using Alpha Shapes.

    This is like "shrink-wrapping" a surface around the points. It's great for
    getting a sense of the object's volume and overall form. The `alpha` value
    controls how tightly the wrap fits. Smaller alpha = tighter, more detailed fit.

    Args:
        pcd: The input Open3D PointCloud.
        alpha: The alpha value for the alpha shape algorithm.

    Returns:
        An Open3D TriangleMesh object.
    """
    # The Alpha Shape algorithm creates a manifold mesh from a point cloud.
    print(f"Creating Alpha Shape mesh with alpha={alpha}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # This gives the mesh a clean, uniform color.
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.compute_vertex_normals()
    
    print(f"Generated mesh with {len(mesh.triangles)} triangles.")
    return mesh
