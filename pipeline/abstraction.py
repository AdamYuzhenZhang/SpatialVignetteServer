"""
abstraction.py - A library of functions for geometric abstraction.

Each function takes in a `ProcessedVignette` object,
extract a specific type of information, and enrich the vignette with that data.
"""

import open3d as o3d
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pyransac3d as pyrsc

# Import our custom data structure
from .vignette_data import ProcessedVignette

# --- Main Pipeline ---

def run_abstraction_pipeline(
    vignette: ProcessedVignette,
    abstractions_to_run: List[str],
    params: Optional[Dict[str, Any]] = None,
    user_guidance: Optional[Dict[str, Any]] = None
) -> ProcessedVignette:
    """
    Runs a series of specified abstraction functions on a vignette.
    """
    params = params or {}
    print(f"--- Running Abstraction Pipeline for: {abstractions_to_run} ---")

    # Pass both the specific parameters and the general user_guidance to each function
    if 'dominant_axes' in abstractions_to_run:
        extract_dominant_axes(vignette, user_guidance=user_guidance, **params.get('dominant_axes', {}))
    
    if 'dominant_planes' in abstractions_to_run:
        extract_dominant_planes(vignette, user_guidance=user_guidance, **params.get('dominant_planes', {}))
    
    if 'structural_lines' in abstractions_to_run:
        extract_structural_lines(vignette, user_guidance=user_guidance, **params.get('structural_lines', {}))

    if 'point_importance' in abstractions_to_run:
        calculate_point_importance(vignette, user_guidance=user_guidance, **params.get('point_importance', {}))

    if 'stylized_colors' in abstractions_to_run:
        stylize_colors(vignette, user_guidance=user_guidance, **params.get('stylized_colors', {}))

    vignette.save() # Perform a final save after all abstractions are run.
    print("--- Abstraction Pipeline Complete ---")
    return vignette


# --- High-Level Abstraction Functions (Modify Metadata) ---

def _compute_obb_features(points: np.ndarray) -> Optional[Dict[str, Any]]:
    """Helper function to compute Oriented Bounding Box features from points."""
    if len(points) < 3:
        return None # Not enough points to form a box
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    try:
        # Uses Principal Component Analysis (PCA) to find the tightest-fitting box around the points.
        obb = pcd.get_oriented_bounding_box()
    except RuntimeError:
        return None

    axes = [obb.R[:, i] for i in range(3)]
    primary_axis_index = np.argmax(obb.extent)
    
    return {
        "center": obb.center.tolist(),
        "axes": [axis.tolist() for axis in axes],
        "extent": obb.extent.tolist(),
        "primary_axis": axes[primary_axis_index].tolist(),
        "point_count": len(points)
    }

def extract_dominant_axes(
    vignette: ProcessedVignette,
    user_guidance: Optional[Dict[str, Any]] = None,
    cluster_eps: float = 0.05,
    min_cluster_points: int = 50
) -> None:
    """
    Finds the dominant structural axes for the whole vignette and its parts.

    This function performs a two-level analysis:
    1. Global Axis: Computes the single Oriented Bounding Box (OBB) for the entire vignette.
    2. Component Axes: Uses DBSCAN clustering to find distinct geometric parts and
       computes an OBB for each individual part.

    Results are stored in the vignette's metadata and a 'component_id' per-point
    attribute is added for visualization and further analysis.

    Args:
        vignette: The vignette to analyze. It will be modified in-place.
        user_guidance: Optional user input. Can contain 'priority_axis' to influence results.
        cluster_eps: The DBSCAN epsilon value for separating components. This is a crucial
                     parameter to tune based on point cloud density.
        min_cluster_points: The minimum number of points required to form a component.
    """
    print("Extracting dominant axes...")
    user_guidance = user_guidance or {}
    
    # Start fresh by clearing any previously computed axes.
    vignette.clear_abstractions('axes', auto_save=False)
    
    # --- 1. Global Axis Analysis ---
    print("   - Analyzing global axis...")
    global_features = _compute_obb_features(vignette.points)
    if global_features:
        global_features['type'] = 'global'
        vignette.add_abstraction('axes', global_features, auto_save=False)
        print(f"   - Global axis found with extent: {np.round(global_features['extent'], 2)}")

    # --- 2. Component Identification via Clustering ---
    print(f"   - Clustering components with eps={cluster_eps}...")
    # DBSCAN is a density-based clustering algorithm. It's great for finding
    # arbitrarily shaped clusters and separating them from noise.
    clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_points).fit(vignette.points)
    labels = clustering.labels_
    
    # Add the cluster labels as a per-point attribute. This is invaluable for
    # debugging and visualization, allowing you to color the vignette by component.
    # Label -1 is designated for noise points by DBSCAN.
    vignette.set_attribute('component_id', labels, auto_save=False)
    
    unique_labels = set(labels)
    num_components = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"   - Found {num_components} components.")

    # --- 3. Per-Component Axis Analysis ---
    # TODO: If a 'focus_point' is in user_guidance, give more weight to components near it.
    for label in unique_labels:
        if label == -1:
            continue # Skip noise points

        # Get the subset of points belonging to this specific component
        component_indices = np.where(labels == label)[0]
        component_points = vignette.points[component_indices]
        
        component_features = _compute_obb_features(component_points)
        
        if component_features:
            component_features['type'] = 'component'
            component_features['component_id'] = int(label) # Ensure JSON-serializable type
            vignette.add_abstraction('axes', component_features, auto_save=False)
    
    print("Finished extracting dominant axes.")



def _extract_primitive(
    vignette: ProcessedVignette,
    primitive_type: str,
    distance_threshold: float,
    min_points: int,
    user_guidance: Optional[Dict[str, Any]] = None
):
    """
    A generic, private helper function to extract any primitive supported by pyransac3d.
    
    This function contains the core iterative RANSAC logic and calculates
    the Oriented Bounding Box (OBB) for the inlier points of each primitive found.
    """
    primitive_map = {
        'plane': pyrsc.Plane,
        'cylinder': pyrsc.Cylinder,
        'sphere': pyrsc.Sphere,
        'cuboid': pyrsc.Cuboid
    }
    
    if primitive_type not in primitive_map:
        raise ValueError(f"Unsupported primitive type: {primitive_type}")

    primitive_class = primitive_map[primitive_type]
    plural_name = f"{primitive_type}s"
    id_name = f"{primitive_type}_id"

    print(f"Extracting dominant {plural_name}...")
    user_guidance = user_guidance or {}
    vignette.clear_abstractions(plural_name, auto_save=False)
    
    labels = np.zeros(vignette.n_points, dtype=int)
    remaining_points = vignette.points.copy()
    remaining_indices = np.arange(vignette.n_points)
    current_id = 1
    
    while len(remaining_points) > min_points:
        primitive_fitter = primitive_class()
        
        inlier_indices_relative = []
        params = None
        
        try:
            if primitive_type == 'plane':
                params, inlier_indices_relative = primitive_fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=1000)
            elif primitive_type == 'cylinder':
                center, axis, radius, inlier_indices_relative = primitive_fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=1000)
                params = (center, axis, radius)
            elif primitive_type == 'sphere':
                center, radius, inlier_indices_relative = primitive_fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=1000)
                params = (center, radius)
            elif primitive_type == 'cuboid':
                _, inlier_indices_relative = primitive_fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=100)

        except (ValueError, RuntimeError) as e:
            print(f"   - RANSAC fit raised an exception for {primitive_type}: {e}")
            break

        if len(inlier_indices_relative) < min_points:
            break
            
        inlier_indices_original = remaining_indices[inlier_indices_relative]
        inlier_points = vignette.points[inlier_indices_original]
        
        abstraction_data = {}
        fit_error = 0.0

        # --- Calculate Fit Error for each primitive type ---
        if primitive_type == 'plane':
            equation = np.array(params)
            # dist = |ax + by + cz + d| / sqrt(a^2+b^2+c^2)
            dists = np.abs(inlier_points @ equation[:3] + equation[3]) / np.linalg.norm(equation[:3])
            fit_error = np.mean(dists)
            abstraction_data.update({"equation": params})

        elif primitive_type == 'cylinder':
            center, axis, radius = np.array(params[0]), np.array(params[1]), params[2]
            # dist = | distance from point to axis - radius |
            vecs = inlier_points - center
            dists_to_axis = np.linalg.norm(np.cross(vecs, axis), axis=1) / np.linalg.norm(axis)
            fit_error = np.mean(np.abs(dists_to_axis - radius))
            abstraction_data.update({"center": center.tolist(), "axis": axis.tolist(), "radius": radius})

        elif primitive_type == 'sphere':
            center, radius = np.array(params[0]), params[1]
            # dist = | distance from point to center - radius |
            dists_to_center = np.linalg.norm(inlier_points - center, axis=1)
            fit_error = np.mean(np.abs(dists_to_center - radius))
            abstraction_data.update({"center": center.tolist(), "radius": radius})

        if len(inlier_points) > 3:
            obb_pcd = o3d.geometry.PointCloud()
            obb_pcd.points = o3d.utility.Vector3dVector(inlier_points)
            obb = obb_pcd.get_oriented_bounding_box()
            
            abstraction_data.update({
                "obb_center": obb.center.tolist(),
                "obb_rotation": obb.R.tolist(),
                "obb_extent": obb.extent.tolist()
            })

            if primitive_type == 'cuboid':
                # Transform inlier points into the OBB's local coordinate system.
                # R.T is the inverse of the rotation matrix.
                points_local = (inlier_points - obb.center) @ obb.R.T
                
                # The half-extents of the box define the face locations.
                half_extents = obb.extent / 2.0
                
                # Calculate the distance of each point from the box's surface.
                # This measures how far a point is "outside" the box along each axis.
                # Points inside the box will have a distance of 0.
                dists_from_surface = np.maximum(0, np.abs(points_local) - half_extents)
                
                # The error for each point is its Euclidean distance from the surface.
                point_errors = np.linalg.norm(dists_from_surface, axis=1)
                
                # The final fit error is the average of these distances.
                fit_error = np.mean(point_errors)
        
        labels[inlier_indices_original] = current_id

        abstraction_data.update({
            id_name: current_id,
            "point_count": len(inlier_indices_original),
            "point_indices": inlier_indices_original.tolist(),
            "fit_error": fit_error # Store the calculated fit error
        })

        vignette.add_abstraction(plural_name, abstraction_data, auto_save=False)
        
        current_id += 1
        remaining_points = np.delete(remaining_points, inlier_indices_relative, axis=0)
        remaining_indices = np.delete(remaining_indices, inlier_indices_relative)
        
    vignette.set_attribute(id_name, labels, auto_save=False)
    print(f"Finished extracting dominant {plural_name}.")

def extract_dominant_planes(vignette: ProcessedVignette, user_guidance: Optional[Dict[str, Any]] = None, distance_threshold: float = 0.01, min_points_per_plane: int = 100) -> None:
    """Finds significant planes using RANSAC and enriches the vignette."""
    _extract_primitive(vignette, 'plane', distance_threshold, min_points_per_plane, user_guidance)

def extract_dominant_cylinders(vignette: ProcessedVignette, user_guidance: Optional[Dict[str, Any]] = None, distance_threshold: float = 0.01, min_points_per_cylinder: int = 100) -> None:
    """Finds significant cylinders using RANSAC and enriches the vignette."""
    _extract_primitive(vignette, 'cylinder', distance_threshold, min_points_per_cylinder, user_guidance)

def extract_dominant_spheres(vignette: ProcessedVignette, user_guidance: Optional[Dict[str, Any]] = None, distance_threshold: float = 0.01, min_points_per_sphere: int = 100) -> None:
    """Finds significant spheres using RANSAC and enriches the vignette."""
    _extract_primitive(vignette, 'sphere', distance_threshold, min_points_per_sphere, user_guidance)

def extract_dominant_cuboids(vignette: ProcessedVignette, user_guidance: Optional[Dict[str, Any]] = None, distance_threshold: float = 0.01, min_points_per_cuboid: int = 100) -> None:
    """Finds significant cuboids using RANSAC and enriches the vignette."""
    _extract_primitive(vignette, 'cuboid', distance_threshold, min_points_per_cuboid, user_guidance)


def compose_best_fit_abstraction(
        vignette: ProcessedVignette, 
        min_coverage_ratio: float = 0.01, 
        default_distance_threshold: float = 0.02
    ) -> None:
    """
    Analyzes all found primitives and composes a "best fit" scene using a more
    intelligent scoring system that balances point coverage with fit quality.
    """
    print("Composing best-fit abstraction...")
    vignette.clear_abstractions('best_fit_composition', auto_save=False)

    all_primitives = []
    candidate_abstractions = vignette.get_abstractions() or {}
    
    for primitive_type, primitives in candidate_abstractions.items():
        if primitive_type not in ['planes', 'cylinders', 'spheres', 'cuboids']:
            continue
        for primitive in primitives:
            primitive['type'] = primitive_type
            
            # --- Balanced Scoring System ---
            point_count = primitive.get('point_count', 0)
            # Use a default error of the RANSAC threshold if not calculated (e.g., for cuboids)
            fit_error = primitive.get('fit_error', default_distance_threshold)
            epsilon = 1e-6 # To prevent division by zero
            
            # The score is higher for more points and a lower fit error.
            primitive['score'] = point_count / (fit_error + epsilon)
            all_primitives.append(primitive)
            
    if not all_primitives:
        print("   - No candidate primitives found to compose. Aborting.")
        return
            
    all_primitives.sort(key=lambda p: p['score'], reverse=True)
    
    unexplained_points_mask = np.ones(vignette.n_points, dtype=bool)
    final_composition = []
    best_fit_labels = np.zeros(vignette.n_points, dtype=int)
    current_best_fit_id = 1
    
    min_points_to_consider = int(vignette.n_points * min_coverage_ratio)

    print(f"   - Starting greedy selection. A primitive must explain at least {min_points_to_consider} new points.")

    for i, primitive in enumerate(all_primitives):
        point_indices = np.array(primitive.get('point_indices', []))
        if len(point_indices) == 0:
            continue

        unexplained_inliers_mask = unexplained_points_mask[point_indices]
        num_newly_explained = np.sum(unexplained_inliers_mask)
        
        primitive_id_key = f"{primitive['type'][:-1]}_id"
        primitive_id = primitive.get(primitive_id_key, 'N/A')

        print(f"     + Evaluating Candidate #{i+1}: {primitive['type']} (ID: {primitive_id}, Score: {primitive['score']:.2f})")
        print(f"     - Point Count: {primitive.get('point_count', 0)}, Fit Error: {primitive.get('fit_error', 'N/A'):.4f}")
        print(f"     - It has {len(point_indices)} total points. Of those, {num_newly_explained} are currently unexplained.")

        if num_newly_explained >= min_points_to_consider:
            print(f"     - ACCEPTED: {num_newly_explained} >= threshold {min_points_to_consider}.")
            
            primitive['best_fit_id'] = current_best_fit_id
            final_composition.append(primitive)
            
            newly_explained_indices = point_indices[unexplained_inliers_mask]
            best_fit_labels[newly_explained_indices] = current_best_fit_id
            
            unexplained_points_mask[newly_explained_indices] = False
            
            current_best_fit_id += 1
        else:
            print(f"     - REJECTED: {num_newly_explained} < threshold {min_points_to_consider}.")
    
    if not final_composition:
        print("\n   - No primitives were selected for the final composition.")
    
    vignette.metadata['best_fit_composition'] = final_composition
    vignette.set_attribute('best_fit_id', best_fit_labels, auto_save=False)
    print("Finished composing best-fit abstraction.")



def extract_structural_lines(
    vignette: ProcessedVignette,
    user_guidance: Optional[Dict[str, Any]] = None,
    curvature_threshold: float = 0.1
) -> None:
    """
    Detects sharp edges and ridges in the point cloud.

    Args:
        vignette: The vignette to analyze. It will be modified in-place.
        user_guidance: Optional user input. Can contain a 'sketch_mask' to guide line detection.
        curvature_threshold: A value to determine what is considered an edge.
    """
    print("Extracting structural lines...")
    user_guidance = user_guidance or {}
    vignette.clear_abstractions('structural_lines', auto_save=False)

    # --- IMPLEMENTATION TO GO HERE ---
    # TODO: If a 'sketch_mask' image is provided in user_guidance, project it onto the
    #       point cloud and increase the likelihood of points under the sketch being edges.

    # 1. Calculate curvature for each point.
    # 2. Threshold points to find edges.
    # 3. Cluster edge points and fit lines.
    # 4. Store line segments in metadata.
    pass

# --- Per-Point Attribute Functions ---

def calculate_point_importance(
    vignette: ProcessedVignette,
    user_guidance: Optional[Dict[str, Any]] = None,
    method: str = 'curvature'
) -> None:
    """
    Generates a per-point 'importance' score.

    Args:
        vignette: The vignette to analyze. It will be modified in-place.
        user_guidance: Optional user input. Can contain 'focus_point' to increase importance nearby.
        method: The method for calculating importance ('curvature', etc.).
    """
    print("Calculating point importance...")
    user_guidance = user_guidance or {}

    # --- IMPLEMENTATION TO GO HERE ---
    # TODO: If 'focus_point' is in user_guidance, add a score bonus to points
    #       based on their proximity to the focus point.

    # 1. Calculate importance score for each point based on the `method`.
    # 2. Normalize scores to [0, 1] range.
    # 3. Use vignette.set_attribute('importance', importance_array, auto_save=False).
    pass

def stylize_colors(
    vignette: ProcessedVignette,
    user_guidance: Optional[Dict[str, Any]] = None,
    style: str = 'sketch'
) -> None:
    """
    Generates a new set of per-point colors based on a style.

    Args:
        vignette: The vignette to analyze. It will be modified in-place.
        user_guidance: Optional user input. Could contain a 'color_palette'.
        style: The style to apply ('sketch', 'monochrome', 'height_ramp', etc.).
    """
    print("Stylizing colors...")
    user_guidance = user_guidance or {}

    # --- IMPLEMENTATION TO GO HERE ---
    # TODO: If 'color_palette' (a list of RGB values) is in user_guidance,
    #       quantize the new colors to match the provided palette.

    # 1. Create a new (N, 3) numpy array for the new colors.
    # 2. Calculate new colors based on the `style`.
    # 3. Use vignette.set_attribute(f'colors_{style}', new_color_array, auto_save=False).
    pass



###### PCA ######

# identify components with DBSCAN
def identify_components(
    vignette: ProcessedVignette,
    cluster_eps: float = 0.05,
    min_cluster_points: int = 50,
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
    
    print(f"   - Found {num_components} components and assigned 'component_id' attribute.")
    return num_components

# PCA
def _compute_pca_properties(points: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Helper to compute PCA and derive high-level structural properties.
    """
    if points.shape[0] < 3:
        return None
        
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Eigenvalues are the variances. Sort them to be safe.
    variances = np.sort(pca.explained_variance_)[::-1]
    l1, l2, l3 = variances
    
    # --- Calculate High-Level Metrics (Normalized 0-1) ---
    # These formulas are standard for 3D shape analysis.
    # We check for l1 > 0 to avoid division by zero on tiny point clusters.
    if l1 > 0:
        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        sphericity = l3 / l1
        anisotropy = (l1 - l3) / l1 # Overall measure of directionality
    else:
        linearity, planarity, sphericity, anisotropy = 0.0, 0.0, 0.0, 0.0

    # Package into a serializable dictionary
    properties = {
        'centroid': pca.mean_.tolist(),
        'axes': pca.components_.tolist(), # Eigenvectors
        'variances': pca.explained_variance_.tolist(), # Eigenvalues
        # High-level abstractions:
        'linearity': linearity,
        'planarity': planarity,
        'sphericity': sphericity,
        'anisotropy': anisotropy,
    }
    return properties


def analyze_structural_properties(vignette: ProcessedVignette, auto_save: bool = False) -> None:
    """
    Analyzes the vignette using PCA to extract high-level structural properties.

    Stores detailed information including centroid, axes (eigenvectors), variances
    (eigenvalues), and derived metrics like linearity, planarity, and sphericity as
    floats between 0 and 1.
    """
    print("Analyzing structural properties with PCA...")
    vignette.clear_abstractions('structural_properties', auto_save=False)
    
    # --- 1. Global Properties ---
    global_props = _compute_pca_properties(vignette.points)
    if global_props:
        global_props['type'] = 'global'
        vignette.add_abstraction('structural_properties', global_props, auto_save=False)
        print(f"   - Global properties: L={global_props['linearity']:.2f}, P={global_props['planarity']:.2f}, S={global_props['sphericity']:.2f}")

    # --- 2. Component Properties ---
    component_labels = vignette.get_attribute('component_id')
    if component_labels is None:
        print("   - Skipping component PCA: 'component_id' not found.")
        return
    unique_labels = set(component_labels)
    # Calculate the actual number of components
    num_components = len(unique_labels)
    if num_components <= 1:
        print(f"   - Skipping per-component analysis ({num_components} component found).")
        return

    print(f"   - Analyzing {num_components} components...")
    for label in sorted(list(unique_labels)):
        if label == -1: continue # Skip noise
        
        component_points = vignette.points[component_labels == label]
        component_props = _compute_pca_properties(component_points)
        
        if component_props:
            component_props['type'] = 'component'
            component_props['component_id'] = int(label)
            print(f"     - Component #{label}: L={component_props['linearity']:.2f}, P={component_props['planarity']:.2f}, S={component_props['sphericity']:.2f}")
            vignette.add_abstraction('structural_properties', component_props, auto_save=False)
            
    if auto_save and vignette.file_path:
        vignette.save()
    print("Finished structural property analysis.")