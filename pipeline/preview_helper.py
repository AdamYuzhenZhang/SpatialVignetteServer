# Helpers for preview in ipynb

import open3d as o3d
import numpy as np
import open3d as o3d
from matplotlib import cm
from typing import List

from pipeline.vignette_data import ProcessedVignette

# Open3d display helper
def display_geometries(geometries: list, window_name: str = "Open3D"):
    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    # Create the window with a custom name
    vis.create_window(window_name=window_name)
    # Add each geometry to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)  
    # Run the visualizer
    vis.run()
    # Destroy it after close
    vis.destroy_window()
    vis.close()
    del vis


# Visualize flow vector
def visualize_vector_attribute(
    vignette: 'ProcessedVignette',
    attribute_name: str,
    step: int = 20,
    scale: float = 0.05
):
    print(f"Visualizing vector attribute: '{attribute_name}'...")
    pcd = vignette.to_open3d() # Get the base point cloud
    vectors = vignette.get_attribute(attribute_name)
    if vectors is None:
        print(f"Error: Attribute '{attribute_name}' not found in vignette.")
        return
    # Subsample points to avoid a cluttered visualization
    points_subset = vignette.points[::step]
    vectors_subset = vectors[::step]
    # Create LineSet geometry
    lines = []
    points_for_lines = []
    for i in range(len(points_subset)):
        p = points_subset[i]
        vec = vectors_subset[i]
        
        # The line starts at the point and ends at point + scaled vector
        start_point = p
        end_point = p + vec * scale
        
        # Add the two points and the line connecting them
        points_for_lines.append(start_point)
        points_for_lines.append(end_point)
        lines.append([2*i, 2*i + 1])
        
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_for_lines),
        lines=o3d.utility.Vector2iVector(lines),
    )
    # Optional: Color the lines
    line_set.paint_uniform_color([1.0, 0.0, 0.0]) # Red lines
    # Draw the original point cloud AND the lines on top
    o3d.visualization.draw_geometries([pcd, line_set])


def _create_axis_visual(
    properties: dict,
    scale_factor: float = 1.0
) -> o3d.geometry.LineSet:
    """Helper to create an Open3D LineSet representing PCA axes."""
    center = np.array(properties['centroid'])
    axes = np.array(properties['axes'])
    variances = np.array(properties['variances'])

    # Axis lengths are proportional to the standard deviation (sqrt of variance)
    axis_lengths = np.sqrt(variances) * scale_factor
    endpoints = center + axes * axis_lengths[:, np.newaxis]

    # Points for the LineSet: center, then the three endpoints
    points = np.vstack([center, endpoints])
    
    # Lines connect the center (index 0) to each endpoint (1, 2, 3)
    lines = [[0, 1], [0, 2], [0, 3]]
    
    # Colors: Red for primary, Green for secondary, Blue for tertiary axis
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def visualize_structural_properties(
    vignette: 'ProcessedVignette',
    axis_scale_factor: float = 1.0
):
    # 1. Generate the point cloud colored by component ID
    try:
        pcd = vignette.to_open3d(color_mode='component_id')
    except AttributeError:
        print("Warning: 'component_id' not found. Displaying with RGB colors.")
        pcd = vignette.to_open3d(color_mode='rgb')

    geometries_to_draw = [pcd]
    
    # 2. Get the structural properties
    all_props = vignette.get_abstractions('structural_properties')
    if not all_props:
        print("No structural properties found to visualize. Run `analyze_structural_properties` first.")
        o3d.visualization.draw_geometries([pcd])
        return

    # 3. Create and add the axis geometries
    print("Generating PCA axis visualizations...")
    for props in all_props:
        axis_visual = _create_axis_visual(props, scale_factor=axis_scale_factor)
        geometries_to_draw.append(axis_visual)
        if props['type'] == 'global':
            print(" - Added GLOBAL axes.")
        else:
            print(f" - Added axes for COMPONENT #{props['component_id']}.")
            
    # 4. Launch the visualizer
    print("\nLaunching Open3D visualizer...")
    o3d.visualization.draw_geometries(geometries_to_draw, window_name="Structural Properties Visualization")




# Visualizing primitives

def _create_plane_meshes(plane_abstractions: List[dict]) -> List[o3d.geometry.TriangleMesh]:
    """Creates a list of two-sided rectangular meshes from plane OBB data."""
    plane_geometries = []
    cmap = cm.get_cmap("Accent")
    
    for i, plane_info in enumerate(plane_abstractions):
        if "obb_center" not in plane_info: continue
        
        center = np.array(plane_info['obb_center'])
        rotation = np.array(plane_info['obb_rotation'])
        extent = np.array(plane_info['obb_extent'])
        
        # Reconstruct a rectangle using the two largest dimensions of the OBB
        sorted_indices = np.argsort(extent)
        major_axis = rotation[:, sorted_indices[2]]
        minor_axis = rotation[:, sorted_indices[1]]
        major_half, minor_half = extent[sorted_indices[2]] / 2.0, extent[sorted_indices[1]] / 2.0

        # Define the four corner vertices of the rectangle
        v1 = center + major_axis * major_half + minor_axis * minor_half
        v2 = center - major_axis * major_half + minor_axis * minor_half
        v3 = center - major_axis * major_half - minor_axis * minor_half
        v4 = center + major_axis * major_half - minor_axis * minor_half
        
        vertices = o3d.utility.Vector3dVector([v1, v2, v3, v4])
        
        # Define triangles for both front and back faces
        triangles = o3d.utility.Vector3iVector([
            [0, 1, 2], [0, 2, 3],  # Front face (e.g., v1-v2-v3 and v1-v3-v4)
            [0, 2, 1], [0, 3, 2]   # Back face (reversed winding order)
        ])
        
        plane_mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        
        # Re-compute normals so both sides are lit correctly
        plane_mesh.compute_vertex_normals()
        plane_mesh.paint_uniform_color(cmap(i / len(plane_abstractions))[:3])
        plane_geometries.append(plane_mesh)
        
    return plane_geometries

def _create_cylinder_meshes(cylinder_abstractions: List[dict], vignette: 'ProcessedVignette') -> List[o3d.geometry.TriangleMesh]:
    """Creates a list of bounded cylinder meshes."""
    cylinder_geometries = []
    cmap = cm.get_cmap("Accent")

    for i, cyl_info in enumerate(cylinder_abstractions):
        ransac_center, axis, radius = np.array(cyl_info['center']), np.array(cyl_info['axis']), cyl_info['radius']
        
        # Project inlier points onto the axis to find the true height and center
        inlier_points = vignette.points[cyl_info['point_indices']]
        projections = np.dot(inlier_points - ransac_center, axis)
        min_proj, max_proj = np.min(projections), np.max(projections)
        height = max_proj - min_proj
        segment_center = ransac_center + axis * ((min_proj + max_proj) / 2.0)
        
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20)
        
        # Robustly calculate the rotation to align the mesh with the cylinder's axis
        z_axis, axis_norm = np.array([0., 0., 1.]), axis / np.linalg.norm(axis)
        rotation_matrix = cylinder_mesh.get_rotation_matrix_from_xyz((0, np.arccos(np.clip(np.dot([1,0,0], [axis_norm[0], axis_norm[1], 0]), -1.0, 1.0)), 0)) @ cylinder_mesh.get_rotation_matrix_from_xyz((0, 0, np.arccos(np.clip(np.dot(z_axis, axis_norm), -1.0, 1.0))))
        
        cylinder_mesh.rotate(rotation_matrix, center=[0,0,0])
        cylinder_mesh.translate(segment_center)
        cylinder_mesh.paint_uniform_color(cmap(i / len(cylinder_abstractions))[:3])
        cylinder_geometries.append(cylinder_mesh)
        
    return cylinder_geometries

def _create_sphere_meshes(sphere_abstractions: List[dict]) -> List[o3d.geometry.TriangleMesh]:
    """Creates a list of sphere meshes."""
    sphere_geometries = []
    cmap = cm.get_cmap("Accent")

    for i, sphere_info in enumerate(sphere_abstractions):
        center, radius = np.array(sphere_info['center']), sphere_info['radius']
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere_mesh.translate(center)
        sphere_mesh.paint_uniform_color(cmap(i / len(sphere_abstractions))[:3])
        sphere_geometries.append(sphere_mesh)

    return sphere_geometries

def _create_cuboid_meshes(cuboid_abstractions: List[dict]) -> List[o3d.geometry.TriangleMesh]:
    """Creates a list of cuboid meshes from OBB data."""
    cuboid_geometries = []
    cmap = cm.get_cmap("Accent")
    
    for i, cuboid_info in enumerate(cuboid_abstractions):
        center, rotation, extent = np.array(cuboid_info['obb_center']), np.array(cuboid_info['obb_rotation']), np.array(cuboid_info['obb_extent'])
        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        cuboid_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb)
        cuboid_mesh.compute_vertex_normals()
        cuboid_mesh.paint_uniform_color(cmap(i / len(cuboid_abstractions))[:3])
        cuboid_geometries.append(cuboid_mesh)

    return cuboid_geometries


def visualize_primitives(vignette: 'ProcessedVignette', primitive_type: str): # ('plane', 'cylinder', 'sphere', 'cuboid')
    """
    Creates an interactive visualization for a specific type of extracted primitive.

    - Colors the point cloud by the primitive's ID.
    - Renders the fitted primitives as distinct colored meshes.
    """
    plural_name = f"{primitive_type}s"
    id_name = f"{primitive_type}_id"

    print(f"--- Visualizing {plural_name.capitalize()} ---")

    # 1. Generate the point cloud colored by the primitive ID
    try:
        pcd = vignette.to_open3d(color_mode=id_name)
        # Points not belonging to any primitive will be colored gray by default
        print(f"   - Coloring points by '{id_name}'.")
    except AttributeError:
        print(f"   - Attribute '{id_name}' not found. Displaying with original colors.")
        pcd = vignette.to_open3d(color_mode='rgb')

    geometries_to_draw = [pcd]
    
    # 2. Get the stored primitive abstractions
    abstractions = vignette.get_abstractions(plural_name)
    if not abstractions:
        print("   - No primitive abstractions found to visualize.")
        o3d.visualization.draw_geometries([pcd], window_name=f"{plural_name.capitalize()} Visualization")
        return
        
    print(f"   - Found {len(abstractions)} primitives to render as meshes.")

    # 3. Dispatch to the correct mesh generation helper
    mesh_geometries = []
    if primitive_type == 'plane':
        mesh_geometries = _create_plane_meshes(abstractions)
    elif primitive_type == 'cylinder':
        mesh_geometries = _create_cylinder_meshes(abstractions, vignette)
    elif primitive_type == 'sphere':
        mesh_geometries = _create_sphere_meshes(abstractions)
    elif primitive_type == 'cuboid':
        mesh_geometries = _create_cuboid_meshes(abstractions)
    
    geometries_to_draw.extend(mesh_geometries)
            
    # 4. Launch the visualizer
    print("\nLaunching Open3D visualizer...")
    o3d.visualization.draw_geometries(geometries_to_draw, window_name=f"{plural_name.capitalize()} Visualization")