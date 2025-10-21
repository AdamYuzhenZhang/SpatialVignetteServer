"""
abstraction.py - A library of functions for geometric abstraction.

Each function takes in a `ProcessedVignette` object,
extract a specific type of information, and enrich the vignette with that data.
"""

import open3d as o3d
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pyransac3d as pyrsc
from pathlib import Path
import cv2
from scipy.spatial import KDTree
import itertools


# Import our custom data structure
from .vignette_data import ProcessedVignette

# --- 2.1 Abstraction from Local 3D Points ---

def analyze_local_features(
    vignette: 'ProcessedVignette',
    search_radius: float = 0.05,
    max_neighbors: int = 30,
    auto_save: bool = False
) -> None:
    # --- (Same setup as before) ---
    print("Analyzing rich local geometric features...")
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(vignette.points)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_neighbors)
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_consistent_tangent_plane(k=max_neighbors)
    vignette.normals = np.asarray(pcd.normals)
    
    print("   - Estimating per-point geometric features...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    n_points = vignette.n_points
    
    # Initialize arrays
    curvatures, edgeness, anisotropy, planarity, sphericity = (np.zeros(n_points) for _ in range(5))
    flow_vectors = np.zeros((n_points, 3)) # NEW: Array to store the flow vectors
    
    for i in range(n_points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], search_radius)
        if k < 5: continue
        
        neighbors = points[idx, :]
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and corresponding eigenvectors
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        l1, l2, l3 = eigenvalues
        v1, _, _ = eigenvectors.T

        # --- NEW: Store the primary eigenvector as the flow vector ---
        flow_vectors[i] = v1

        # (Rest of the calculations for scalar features are the same)
        # ...
        sum_eigenvalues = l1 + l2 + l3
        if sum_eigenvalues < 1e-9: continue
        curvatures[i] = l3 / sum_eigenvalues
        if l1 > 1e-9:
            edgeness[i] = (l1 - l2) / l1
            anisotropy[i] = (l1 - l3) / l1
            planarity[i] = (l2 - l3) / l1
            sphericity[i] = l3 / l1
            
    # Store all attributes
    vignette.set_attribute('curvature', curvatures, auto_save=False)
    vignette.set_attribute('edgeness', edgeness, auto_save=False)
    vignette.set_attribute('anisotropy', anisotropy, auto_save=False)
    vignette.set_attribute('planarity', planarity, auto_save=False)
    vignette.set_attribute('sphericity', sphericity, auto_save=False)
    vignette.set_attribute('flow_vectors', flow_vectors, auto_save=False) # NEW
    
    if auto_save and vignette.file_path: vignette.save()
    print("Finished analyzing local features.")


# --- 2.2 Abstraction from 2D RGB Image ---

# Creating 2D processed images

'''
def generate_edge_map(
    rgb_path: str, 
    mask_path: str, 
    output_path: str, 
    canny_thresh1: int = 100, 
    canny_thresh2: int = 200,
    dilation_kernel_size: int = 3, # Size for dilation
    dilation_iterations: int = 1   # Times applied
) -> str:
    """
    Generates a binary edge map and makes the edges thicker using dilation.
    """
    print(f"Generating edge map -> {Path(output_path).name}")
    rgb_img = cv2.imread(rgb_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask_img)
    
    # Step 1: Find the thin Canny edges
    edges = cv2.Canny(masked_gray, canny_thresh1, canny_thresh2)
    
    # Step 2: Thicken the edges using dilation
    if dilation_kernel_size > 0:
        print(f"   - Dilating edges with kernel size {dilation_kernel_size}x{dilation_kernel_size}...")
        # A simple square kernel
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    else:
        dilated_edges = edges
    
    cv2.imwrite(output_path, dilated_edges)
    return output_path
'''
# New version removing the edge of mask
import cv2
import numpy as np
from pathlib import Path

def generate_edge_map(
    rgb_path: str,
    mask_path: str,
    output_path: str,
    canny_thresh1: int = 100,
    canny_thresh2: int = 200,
    mask_erosion_kernel_size: int = 5,
    dilation_kernel_size: int = 3,
    dilation_iterations: int = 1
) -> str:
    """
    Generates a binary edge map, then uses an eroded mask to remove the
    unwanted boundary edge from the result.

    Args:
        ... (args are the same)
        mask_erosion_kernel_size: The size of the kernel to erode the mask. This
                                  controls how much of the border is removed.
                                  Must be odd. Set to 0 to disable.
    """
    print(f"Generating edge map and removing boundary -> {Path(output_path).name}")
    rgb_img = cv2.imread(rgb_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # --- 1. Generate All Edges First ---
    # Apply the ORIGINAL mask to get all edges, including the boundary.
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask_img)
    all_edges = cv2.Canny(masked_gray, canny_thresh1, canny_thresh2)

    # --- 2. Create the "Safe Zone" Mask ---
    # Erode the original mask to create a new mask that is slightly smaller.
    # This eroded mask defines the area where we want to KEEP edges.
    if mask_erosion_kernel_size > 0:
        print(f"   - Eroding mask with kernel size {mask_erosion_kernel_size}x{mask_erosion_kernel_size} to create safe zone...")
        erosion_kernel = np.ones((mask_erosion_kernel_size, mask_erosion_kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask_img, erosion_kernel, iterations=1)
    else:
        eroded_mask = mask_img

    # --- 3. Filter The Edges ---
    # Use the eroded mask as a "cookie cutter" to remove any edges that
    # fall outside the safe zone. This effectively removes the boundary edge.
    internal_edges = cv2.bitwise_and(all_edges, all_edges, mask=eroded_mask)

    # --- 4. Thicken the Remaining Edges (Optional) ---
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        final_edges = cv2.dilate(internal_edges, kernel, iterations=dilation_iterations)
    else:
        final_edges = internal_edges

    cv2.imwrite(output_path, final_edges)
    return output_path

def generate_detail_map(rgb_path: str, mask_path: str, output_path: str) -> str:
    """
    Generates a grayscale map representing high-frequency texture detail.
    """
    print(f"Generating detail map -> {Path(output_path).name}")
    rgb_img = cv2.imread(rgb_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask_img)

    laplacian_img = np.abs(cv2.Laplacian(masked_gray, cv2.CV_64F))
    
    # Normalize to 0-255 range to save as a visible grayscale image
    if laplacian_img.max() > 0:
        normalized_img = cv2.normalize(laplacian_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        normalized_img = np.zeros_like(laplacian_img, dtype=np.uint8)

    cv2.imwrite(output_path, normalized_img)
    return output_path


def generate_probability_map_from_detail(
    rgb_path: str,
    mask_path: str,
    output_path: str,
    ksize_laplacian: int = 5,
    contrast_power: float = 0.5,
    blur_ksize: int = 31
) -> str:
    """
    Generates a smoothed, high-contrast grayscale map representing the "strength of detail,"
    ideal for use as a probability map.

    Args:
        rgb_path: Path to the source RGB image.
        mask_path: Path to the mask image.
        output_path: The file path to save the final probability map.
        ksize_laplacian: The kernel size for the Laplacian filter. A smaller value
                         (e.g., 3) is more sensitive to fine noise. A larger value
                         (e.g., 7) captures more significant edges. Must be odd.
        contrast_power: A power to apply for contrast enhancement. Values < 1.0
                        (e.g., 0.5) will brighten faint details. Values > 1.0 will
                        make only the strongest details stand out.
        blur_ksize: The kernel size for the Gaussian blur. This smooths the map
                    into broader regions, making it less noisy. Must be odd.

    Returns:
        The path to the saved probability map image.
    """
    print(f"Generating probability map from detail -> {Path(output_path).name}")
    rgb_img = cv2.imread(rgb_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # --- 1. Calculate Raw Detail with Tunable Kernel ---
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask_img)

    # Use the tunable ksize for the Laplacian filter
    laplacian_img = np.abs(cv2.Laplacian(masked_gray, cv2.CV_64F, ksize=ksize_laplacian))
    
    # --- 2. Normalize, Amplify, and Smooth the Map ---
    
    # a) Normalize the raw detail map to a floating-point range of 0-1
    float_map = np.zeros_like(laplacian_img, dtype=np.float64)
    if laplacian_img.max() > 0:
        float_map = cv2.normalize(laplacian_img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    # b) Amplify the signal using a power function for contrast
    # This is key to making faint details more visible
    contrast_map = np.power(float_map, contrast_power)
    
    # c) Smooth the map to create broader, less noisy regions
    # 
    if blur_ksize > 0:
        blurred_map = cv2.GaussianBlur(contrast_map, (blur_ksize, blur_ksize), 0)
    else:
        blurred_map = contrast_map
        
    # --- 3. Finalize and Save the Processed Map ---
    
    # Re-apply the mask to ensure the background is pure black after blurring
    final_map_float = cv2.bitwise_and(blurred_map, blurred_map, mask=mask_img)

    # Normalize the final 0-1 float map to a 0-255 uint8 range for saving as an image
    if final_map_float.max() > 0:
        final_map_uint8 = cv2.normalize(final_map_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        final_map_uint8 = np.zeros_like(final_map_float, dtype=np.uint8)

    cv2.imwrite(output_path, final_map_uint8)
    print(f"   - Saved smoothed probability map to {output_path}")
    return output_path

def generate_stylized_image(
    rgb_path: str, 
    mask_path: str, 
    output_path: str, 
    k: int
) -> str:
    """
    Generates a stylized version of the image using K-Means color quantization,
    intelligently analyzing ONLY the pixels within the provided mask.
    """
    print(f"Generating {k}-color stylized image -> {Path(output_path).name}")
    rgb_img = cv2.imread(rgb_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # --- 1. Isolate Object Pixels for Analysis ---
    # Create a boolean mask where True corresponds to the object.
    # We use > 0 to be safe, but > 127 is also common for binary masks.
    object_mask = mask_img > 0

    # Use the boolean mask to select only the pixels from the RGB image
    # that are part of the object.
    object_pixels = rgb_img[object_mask]
    
    # If there are no pixels in the mask, return a black image.
    if object_pixels.shape[0] == 0:
        print("   [Warning] No object pixels found in the mask. Output will be black.")
        cv2.imwrite(output_path, np.zeros_like(rgb_img))
        return output_path

    # Reshape for k-means and convert to float32
    pixels_for_kmeans = object_pixels.reshape((-1, 3)).astype(np.float32)

    # --- 2. Run K-Means ONLY on Object Pixels ---
    # The algorithm now only sees the true colors of the object.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels_for_kmeans, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    # The result contains the new stylized colors for ONLY the object pixels.
    stylized_object_pixels = centers[labels.flatten()]

    # --- 3. Reconstruct the Final Image ---
    # Start with a completely black image that has the same dimensions as the original.
    stylized_img = np.zeros_like(rgb_img)

    # Use the same boolean mask to place the stylized object pixels
    # back into their correct locations on the black canvas.
    stylized_img[object_mask] = stylized_object_pixels.reshape((-1, 3))
    
    cv2.imwrite(output_path, stylized_img)
    return output_path

'''
# Applying 2D images to 3D points
def apply_feature_map_to_vignette(
    vignette: 'ProcessedVignette',
    feature_map_path: str,
    attribute_name: str,
    auto_save: bool = False
) -> 'ProcessedVignette':
    """
    Projects 3D points to a 2D feature map and samples the values.
    """
    print(f"Applying feature map '{Path(feature_map_path).name}' as attribute '{attribute_name}'...")
    feature_img = cv2.imread(feature_map_path, cv2.IMREAD_UNCHANGED)
    if feature_img is None:
        print(f"   [ERROR] Could not load feature map from {feature_map_path}")
        return vignette

    # 1. Retrieve necessary metadata for projection
    metadata = vignette.metadata.get('capture_metadata', {})
    center_offset = np.array(metadata.get('center_offset'))
    intrinsics_data = metadata.get('camera_intrinsics')
    depth_res = metadata.get('resolution')

    if center_offset is None or intrinsics_data is None or depth_res is None:
        print("   [ERROR] Missing 'center_offset', 'camera_intrinsics', or 'resolution' in metadata.")
        return vignette

    # 2. Reconstruct the scaled intrinsic matrix used during point generation
    intrinsics_matrix = np.array(intrinsics_data['columns']).T
    depth_width, depth_height = depth_res[0], depth_res[1]
    
    # NOTE: Assuming the feature map has the same dimensions as the depth map.
    # The original color image size isn't needed here.
    img_height, img_width = feature_img.shape[:2]
    x_scale, y_scale = depth_width / img_width, depth_height / img_height
    fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    K = np.array([
        [fx * x_scale, 0, cx * x_scale],
        [0, fy * y_scale, cy * y_scale],
        [0, 0, 1]
    ])

    # 3. Project points back to 2D
    # First, undo the centering translation to get points in camera coordinates
    points_camera = vignette.points + center_offset
    
    # Project from camera space to image space
    points_proj = (K @ points_camera.T).T
    
    # Normalize to get pixel coordinates (u, v)
    valid_mask = points_proj[:, 2] > 0
    u = (points_proj[:, 0] / points_proj[:, 2]).astype(int)
    v = (points_proj[:, 1] / points_proj[:, 2]).astype(int)
    
    bounds_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    final_mask = valid_mask & bounds_mask
    
    # 4. Sample the feature image and set the attribute
    if feature_img.ndim == 3 and feature_img.shape[2] == 3: # Color image
        sampled_values = np.zeros((vignette.n_points, 3))
        # Convert BGR to RGB and normalize 0-255 to 0-1
        sampled_colors = cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB) / 255.0
        sampled_values[final_mask] = sampled_colors[v[final_mask], u[final_mask]]
    else: # Grayscale image
        sampled_values = np.zeros(vignette.n_points)
        # Normalize 0-255 to 0-1
        normalized_features = feature_img.astype(float) / 255.0
        sampled_values[final_mask] = normalized_features[v[final_mask], u[final_mask]]
        
    # Set the new attribute regardless of its shape
    vignette.set_attribute(attribute_name, sampled_values, auto_save=auto_save)
    print(f"   - Set attribute '{attribute_name}' with shape {sampled_values.shape}.")
    return vignette
'''

# Using stored UV here
def apply_feature_map_to_vignette(
    vignette: 'ProcessedVignette',
    feature_map_path: str,
    attribute_name: str
) -> 'ProcessedVignette':
    """
    Applies a 2D feature map to a vignette by sampling it using the
    vignette's pre-stored UV coordinates.
    """
    print(f"Applying feature map '{Path(feature_map_path).name}' using stored UVs...")
    
    # 1. Load the feature map
    feature_img = cv2.imread(feature_map_path, cv2.IMREAD_UNCHANGED)
    if feature_img is None:
        raise IOError(f"Could not load feature map from {feature_map_path}")
    
    img_height, img_width = feature_img.shape[:2]

    # 2. Get the stored UV coordinates from the vignette
    uv_coords = vignette.get_attribute('uv_coords')
    if uv_coords is None:
        raise ValueError("Vignette is missing the 'uv_coords' attribute. Please regenerate it using the new creation script.")

    # 3. De-normalize the UVs to get pixel coordinates in the feature map
    # We multiply by (size - 1) to correctly map back to 0-indexed pixels
    u_pixel = (uv_coords[:, 0] * (img_width - 1)).astype(int)
    v_pixel = (uv_coords[:, 1] * (img_height - 1)).astype(int)

    # 4. Sample the feature image at the calculated pixel coordinates
    if feature_img.ndim == 3: # Color image
        sampled_values = np.zeros((vignette.n_points, 3), dtype=np.float64)
        sampled_colors_bgr = feature_img[v_pixel, u_pixel]
        sampled_values = cv2.cvtColor(sampled_colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    else: # Grayscale image
        sampled_values = feature_img[v_pixel, u_pixel].astype(np.float64) / 255.0

    # 5. Set the new attribute
    vignette.set_attribute(attribute_name, sampled_values)
    return vignette

# extract color palette from rgb image
def extract_color_palette(
    vignette: 'ProcessedVignette',
    rgb_path: str,
    mask_path: str,
    k: int = 8,
    output_dir: Optional[str] = None,
    auto_save: bool = False
) -> 'ProcessedVignette':
    """
    Extracts a dominant color palette from the masked RGB image and stores it
    in the vignette's metadata.

    Args:
        vignette: The vignette to analyze.
        rgb_path: Path to the source RGB image.
        mask_path: Path to the binary mask of the object.
        k: The number of dominant colors to extract for the palette.
        output_dir: If provided, saves a visualization of the palette as a PNG.

    Returns:
        The modified vignette with the new color_palette metadata.
    """
    print(f"Extracting {k}-color palette...")
    try:
        rgb_img = cv2.imread(rgb_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if rgb_img is None or mask_img is None:
            raise IOError("Could not load images.")
    except Exception as e:
        print(f"   [ERROR] Could not load images: {e}")
        return vignette

    # 1. Isolate the pixels of the object using the mask
    # Find all coordinates where the mask is white
    object_pixels_coords = np.where(mask_img > 127)
    # Get the BGR values for only those pixels
    object_pixels = rgb_img[object_pixels_coords].astype(np.float32)

    if len(object_pixels) < k:
        print(f"   [WARNING] Fewer pixels ({len(object_pixels)}) than k ({k}). Skipping.")
        return vignette

    # 2. Run K-Means clustering to find the dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(object_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 3. Calculate the proportion of each color
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(object_pixels)
    
    # 4. Structure the data for storage
    # Sort by proportion, from most dominant to least
    sort_indices = np.argsort(proportions)[::-1]
    sorted_centers = centers[sort_indices]
    sorted_proportions = proportions[sort_indices]

    # Convert BGR centers to RGB and ensure they are JSON-serializable integers
    colors_rgb_255 = [c[::-1].astype(int).tolist() for c in sorted_centers]

    palette_data = {
        "k": k,
        "colors_rgb_255": colors_rgb_255,
        "proportions": sorted_proportions.tolist()
    }
    
    # 5. Store the palette in the vignette's metadata using our new helper
    vignette.set_metadata_property('color_palette', palette_data, auto_save=auto_save)

    # 6. (Optional) Create and save a visualization of the palette
    if output_dir:
        palette_vis_path = Path(output_dir) / "palette.png"
        bar_height = 100
        bar_width = 500
        palette_img = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
        current_x = 0
        for color, prop in zip(colors_rgb_255, sorted_proportions):
            w = int(prop * bar_width)
            # OpenCV uses BGR, so we convert our RGB back
            cv2.rectangle(palette_img, (current_x, 0), (current_x + w, bar_height), color[::-1], -1)
            current_x += w
        cv2.imwrite(str(palette_vis_path), palette_img)
        print(f"Saved palette visualization to {palette_vis_path}")

    return vignette


# --- 2.3 Abstraction from 3D Points ---

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
    try:
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

        print(f"   - Analyzing {num_components - 1} components...")
        for label in sorted(list(unique_labels)):
            if label == -1: continue # Skip noise
            
            component_points = vignette.points[component_labels == label]
            component_props = _compute_pca_properties(component_points)
            
            if component_props:
                component_props['type'] = 'component'
                component_props['component_id'] = int(label)
                print(f"     - Component #{label}: L={component_props['linearity']:.2f}, P={component_props['planarity']:.2f}, S={component_props['sphericity']:.2f}")
                vignette.add_abstraction('structural_properties', component_props, auto_save=False)
        
    finally:
        if auto_save and vignette.file_path:
            vignette.save()
        print("Finished structural property analysis.")

# pyransac3d primitive

def _fit_primitive_to_subset(
    vignette: 'ProcessedVignette',
    primitive_type: str, # plane, cylinder, sphere, or cuboid
    attribute_name: str, # per-point attribute, like component_id
    attribute_value: Any, # target value for per-point attribute
    distance_threshold: float, # RANSAC distance threshold
    min_points: int # min number of inliers
) -> List[Tuple[Any, np.ndarray]]:
    """
    Fits primitives to a subset of points defined by an attribute filter.
    Returns:
        A list of tuples, where each tuple contains (parameters, global_inlier_indices).
    """
    primitive_map = {'plane': pyrsc.Plane, 'cylinder': pyrsc.Cylinder, 'sphere': pyrsc.Sphere, 'cuboid': pyrsc.Cuboid}
    primitive_class = primitive_map[primitive_type]
    
    # Filter by labels
    labels = vignette.get_attribute(attribute_name)
    if labels is None: return []
    
    global_subset_indices = np.where(labels == attribute_value)[0]
    if len(global_subset_indices) < min_points: return []
        
    subset_points = vignette.points[global_subset_indices]
    
    # Run iterative RANSAC on this subset of points
    found_primitives = []
    remaining_points = subset_points.copy()
    relative_indices = np.arange(len(remaining_points)) # Indices relative to the subset

    while len(remaining_points) > min_points:
        fitter = primitive_class()
        params, inlier_indices_local = None, []

        try:
            # different return values from pyransac3d's fit methods
            if primitive_type == 'plane':
                params, inlier_indices_local = fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=1000)
            elif primitive_type == 'cylinder':
                center, axis, radius, inlier_indices_local = fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=1000)
                params = (center, axis, radius)
            elif primitive_type == 'sphere':
                center, radius, inlier_indices_local = fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=1000)
                params = (center, radius)
            elif primitive_type == 'cuboid':
                cuboid_obj, inlier_indices_local = fitter.fit(remaining_points, thresh=distance_threshold, maxIteration=100)
                params = cuboid_obj # Store the object itself
        except (ValueError, RuntimeError):
            break

        if len(inlier_indices_local) < min_points:
            break
            
        # 3. Map the local inlier indices back to global vignette indices
        # This is the most crucial step for composing results later.
        inlier_indices_in_subset = relative_indices[inlier_indices_local]
        global_inlier_indices = global_subset_indices[inlier_indices_in_subset]
        
        found_primitives.append((params, global_inlier_indices))
        
        remaining_points = np.delete(remaining_points, inlier_indices_local, axis=0)
        relative_indices = np.delete(relative_indices, inlier_indices_local, axis=0)
        
    return found_primitives


# This one uses region growing segmentation instead of ransac


def _fit_plane_with_pca(cluster_points: np.ndarray) -> np.ndarray:
    """
    Fits a plane to a set of points using PCA (least-squares fit).
    Assumes all points are inliers.
    Returns: The plane equation as a 4-element numpy array [a, b, c, d].
    """
    pca = PCA(n_components=3)
    pca.fit(cluster_points)
    # The normal of the plane is the eigenvector with the smallest eigenvalue
    normal = pca.components_[2]
    # The centroid of the points lies on the plane
    centroid = pca.mean_
    # Calculate d from the plane equation ax + by + cz + d = 0
    # d = -(a*x_c + b*y_c + c*z_c) = -dot(normal, centroid)
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d])

def _split_by_offset_along_normal(
    cluster_points: np.ndarray,
    n_hat: np.ndarray,
    plane_dist_eps: float,
    min_points_per_subcluster: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Split into parallel layers with 1-D DBSCAN on s = n·x.

    Returns
    -------
    subclusters : list[np.ndarray]  # local indices for each kept subcluster
    noise_idx  : np.ndarray         # local indices labeled as noise by DBSCAN
    s          : np.ndarray         # 1-D coordinates used for splitting (N,)
    """
    print(f"     [DEBUG]      - Splitting {len(cluster_points)} points by offset along normal...")
    print(f"     [DEBUG]        - Using 1D DBSCAN on s=n·x with eps={plane_dist_eps:.6f}, min_pts={min_points_per_subcluster}")

    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    s = (cluster_points @ n_hat).reshape(-1, 1)

    labels = DBSCAN(eps=plane_dist_eps, min_samples=min_points_per_subcluster).fit(s).labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    noise_cnt = int(counts[unique_labels == -1][0]) if -1 in unique_labels else 0
    kept = 0

    print(f"     [DEBUG]        - 1D DBSCAN labels: {list(unique_labels)} (noise=-1)")
    print(f"     [DEBUG]        - Noise points in 1D: {noise_cnt}")

    subclusters: list[np.ndarray] = []
    for lab in unique_labels:
        if lab == -1: 
            continue
        idx = np.where(labels == lab)[0]
        if idx.size >= min_points_per_subcluster:
            subclusters.append(idx)
            kept += 1
            print(f"     [DEBUG]        - Kept subcluster #{int(lab)} with {idx.size} points.")
        else:
            print(f"     [DEBUG]        - Discarded small subcluster #{int(lab)} with {idx.size} points.")

    noise_idx = np.where(labels == -1)[0]
    return subclusters, noise_idx, s.ravel()

def _assign_noise_points_to_layers(
    s: np.ndarray,
    noise_idx: np.ndarray,
    layer_indices: list[np.ndarray],
    plane_distance_eps: float,
    strategy: str  # "force_assign" | "merge"
) -> list[np.ndarray]:
    """
    Post-process noise by attaching to nearest layer centers in 1-D s space.

    - "force_assign": assign ALL noise to nearest layer (no distance cutoff).
    - "merge": assign ONLY if within plane_distance_eps; others remain noise.
    """
    if len(noise_idx) == 0 or len(layer_indices) == 0:
        return layer_indices

    # Compute layer centers in s
    layer_means = np.array([s[idx].mean() for idx in layer_indices])
    noise_s = s[noise_idx]

    # For each noise point, find nearest layer center
    # shape: (num_noise, num_layers)
    dists = np.abs(noise_s[:, None] - layer_means[None, :])
    nearest = dists.argmin(axis=1)
    nearest_dist = dists[np.arange(len(noise_idx)), nearest]

    # Decide which noise points to attach
    if strategy == "force_assign":
        attach_mask = np.ones_like(nearest_dist, dtype=bool)
    elif strategy == "merge":
        attach_mask = nearest_dist <= plane_distance_eps
    else:
        attach_mask = np.zeros_like(nearest_dist, dtype=bool)

    # Attach
    for k, attach in enumerate(attach_mask):
        if attach:
            layer_indices[nearest[k]] = np.concatenate([layer_indices[nearest[k]], [noise_idx[k]]])

    kept = int(attach_mask.sum())
    dropped = int((~attach_mask).sum())
    print(f"     [DEBUG]        - Noise attach summary [{strategy}]: kept={kept}, left_unassigned={dropped}")
    return layer_indices

def _make_new_plane_from_noise_if_large(
    points_local: np.ndarray,
    noise_idx: np.ndarray,
    min_points_per_plane: int
) -> list[np.ndarray]:
    """
    If the noise set is large enough, turn it into its own subcluster.
    """
    if noise_idx.size >= min_points_per_plane:
        print(f"     [DEBUG]        - Promoting noise to new subcluster with {noise_idx.size} points.")
        return [noise_idx]
    return []

'''
def _region_growing_plane_segmentation(
    points_subset: np.ndarray,
    global_indices_of_subset: np.ndarray,
    *,
    normal_search_radius: float,
    normal_max_nn: int = 30,
    normal_angle_tolerance_deg: float = 15.0,
    min_samples_normals_ratio: float = 0.10,
    plane_distance_eps: float,
    min_points_per_plane: int,
    refinement_method: str = "pca",
    ransac_max_iterations: int = 500,
    # NEW:
    noise_strategy_layer: str = "force_assign"  # "force_assign" | "merge" | "new_plane" | "keep"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Same as before, but now we ensure group-level inclusion by handling 1-D 'noise'.
    """
    N = len(points_subset)
    print(f"     [DEBUG] --- Region Growing (planes) on {N} points ---")
    if N < max(3, min_points_per_plane):
        print("     [DEBUG] Too few points; aborting.")
        return []

    # 1) Normals
    print("     [DEBUG] [1/5] Estimating normals...")
    print(f"     [DEBUG]        - normal_search_radius: {normal_search_radius:.6f}")
    print(f"     [DEBUG]        - normal_max_nn: {normal_max_nn}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_subset)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_search_radius, max_nn=normal_max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=min(30, N))
    normals = np.asarray(pcd.normals, dtype=float)

    # 2) Canonicalize normals
    print("     [DEBUG] [2/5] Canonicalizing normal directions...")
    normals_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    ref = normals_unit.mean(axis=0); ref /= (np.linalg.norm(ref) + 1e-12)
    flip = (normals_unit @ ref) < 0.0
    normals_unit[flip] *= -1.0
    print(f"     [DEBUG]        - Flipped {int(flip.sum())} normals.")

    # 3) Orientation DBSCAN
    print("     [DEBUG] [3/5] Orientation clustering (DBSCAN on unit normals)...")
    eps_normals = 2.0 * np.sin(np.deg2rad(normal_angle_tolerance_deg) / 2.0)
    min_samples_normals = max(min_points_per_plane, int(N * float(min_samples_normals_ratio)))
    print(f"     [DEBUG]        - angle_tol_deg={normal_angle_tolerance_deg:.3f} -> eps_normals={eps_normals:.6f}")
    print(f"     [DEBUG]        - min_samples_normals={min_samples_normals}")
    labels_orient = DBSCAN(eps=eps_normals, min_samples=min_samples_normals).fit(normals_unit).labels_
    unique_labels = np.unique(labels_orient)
    print(f"     [DEBUG]        - Unique orient labels: {list(unique_labels)} (noise=-1)")

    found: list[tuple[np.ndarray, np.ndarray]] = []

    # 4) For each orientation group
    for lab in unique_labels:
        if lab == -1:
            # We could optionally bring orientation noise into a nearest group,
            # but your issue is primarily at the 1-D split; keep it simple here.
            continue

        group_idx_local = np.where(labels_orient == lab)[0]
        if group_idx_local.size < min_points_per_plane:
            print(f"\n     [DEBUG] -> Skip orient group #{int(lab)}: {group_idx_local.size} pts (< min).")
            continue

        group_pts = points_subset[group_idx_local]
        print(f"\n     [DEBUG] -> Orientation group #{int(lab)} | size={group_pts.shape[0]}")

        # PCA normal for projection
        pca = PCA(n_components=3); pca.fit(group_pts)
        proj_normal = pca.components_[2]; proj_normal /= (np.linalg.norm(proj_normal) + 1e-12)
        print(f"     [DEBUG]      - Projection normal (unit): {proj_normal}")

        # 1-D split
        subclusters_local, noise_idx_local, s_vals = _split_by_offset_along_normal(
            group_pts, proj_normal, plane_distance_eps, min_points_per_plane
        )

        # Deal with noise according to strategy
        if noise_strategy_layer == "new_plane":
            # Try promote noise to a plane if big enough; otherwise attach to nearest layer
            new_from_noise = _make_new_plane_from_noise_if_large(group_pts, noise_idx_local, min_points_per_plane)
            if len(new_from_noise) > 0:
                subclusters_local.extend(new_from_noise)
                noise_idx_local = np.array([], dtype=int)
            # Attach any remaining noise (if any) to nearest layers within eps
            subclusters_local = _assign_noise_points_to_layers(
                s_vals, noise_idx_local, subclusters_local, plane_distance_eps, "merge"
            )
        elif noise_strategy_layer in ("force_assign", "merge"):
            subclusters_local = _assign_noise_points_to_layers(
                s_vals, noise_idx_local, subclusters_local, plane_distance_eps, noise_strategy_layer
            )
        elif noise_strategy_layer == "keep":
            print("     [DEBUG]        - Keeping noise unassigned (may be dropped later).")
        else:
            print(f"     [DEBUG]        - Unknown noise_strategy_layer '{noise_strategy_layer}', defaulting to 'keep'.")

        print(f"     [DEBUG]      - Final subcluster count (after noise handling): {len(subclusters_local)}")

        # Refine & emit each subcluster
        for i, sub_idx_local in enumerate(subclusters_local):
            sub_pts = group_pts[sub_idx_local]
            print(f"     [DEBUG]         -> Refining subcluster #{i} (size={len(sub_pts)}) with '{refinement_method}'")

            if refinement_method.lower() == "pca":
                plane_eq = _fit_plane_with_pca(sub_pts)
                inlier_local = np.arange(len(sub_pts), dtype=int)
            else:
                # RANSAC
                try:
                    import pyransac3d as pyrsc
                    fitter = pyrsc.Plane()
                    plane_eq, inlier_local = fitter.fit(sub_pts, thresh=plane_distance_eps, maxIteration=ransac_max_iterations)
                    inlier_local = np.asarray(inlier_local, dtype=int)
                    n = plane_eq[:3]; plane_eq = np.array([*(n / (np.linalg.norm(n)+1e-12)), plane_eq[3]], dtype=float)
                    print(f"     [DEBUG]            - RANSAC inliers: {len(inlier_local)} / {len(sub_pts)}")
                except Exception:
                    print("     [DEBUG]            - RANSAC unavailable/failure; fallback to PCA.")
                    plane_eq = _fit_plane_with_pca(sub_pts)
                    inlier_local = np.arange(len(sub_pts), dtype=int)

            if inlier_local.size < min_points_per_plane:
                print("     [DEBUG]            - Discarded (too few inliers).")
                continue

            # Map local → global
            final_local_in_group = sub_idx_local[inlier_local]
            final_local_in_subset = group_idx_local[final_local_in_group]
            global_inliers = global_indices_of_subset[final_local_in_subset]

            found.append((plane_eq, global_inliers))
            print("     [DEBUG]            - STORED plane with "
                  f"{len(global_inliers)} inliers | eq = "
                  f"[{plane_eq[0]:.6f}, {plane_eq[1]:.6f}, {plane_eq[2]:.6f}, {plane_eq[3]:.6f}]")

    if not found:
        print("\n     [DEBUG] No planes found in this subset.")
    else:
        print(f"\n     [DEBUG] Total planes found in this subset: {len(found)}")
    return found
'''

def _region_growing_plane_segmentation(
    points_subset: np.ndarray,
    global_indices_of_subset: np.ndarray,
    *,
    normal_search_radius: float,
    normal_max_nn: int = 30,
    normal_angle_tolerance_deg: float = 15.0,
    min_samples_normals_ratio: float = 0.10,
    plane_distance_eps: float,
    min_points_per_plane: int,
    refinement_method: str = "pca",
    ransac_max_iterations: int = 500,
    noise_strategy_layer: str = "force_assign",
    assign_unassigned_to_closest_plane: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Performs region-growing based plane segmentation with an optional final pass
    to assign noise points to their closest valid plane.
    """
    N = len(points_subset)
    if N < max(3, min_points_per_plane):
        return []

    # 1) Normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_subset)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_search_radius, max_nn=normal_max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=min(30, N))
    normals = np.asarray(pcd.normals, dtype=np.float64)

    # 2) Canonicalize normals to point in a consistent direction
    normals_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    ref_normal = np.mean(normals_unit, axis=0)
    ref_normal /= (np.linalg.norm(ref_normal) + 1e-12)
    flip_mask = (normals_unit @ ref_normal) < 0.0
    normals_unit[flip_mask] *= -1.0

    # 3) Orientation clustering (DBSCAN on unit normals)
    eps_normals = 2.0 * np.sin(np.deg2rad(normal_angle_tolerance_deg) / 2.0)
    min_samples_normals = max(min_points_per_plane, int(N * float(min_samples_normals_ratio)))
    labels_orient = DBSCAN(eps=eps_normals, min_samples=min_samples_normals).fit(normals_unit).labels_
    unique_labels = np.unique(labels_orient)

    found_planes: list[tuple[np.ndarray, np.ndarray]] = []

    # 4) For each orientation group, find parallel planes
    for label in unique_labels:
        if label == -1:
            continue

        group_idx_local = np.where(labels_orient == label)[0]
        if group_idx_local.size < min_points_per_plane:
            continue

        group_pts = points_subset[group_idx_local]
        
        # Use PCA to get the robust average normal for this orientation group
        pca = PCA(n_components=3)
        pca.fit(group_pts)
        proj_normal = pca.components_[2]

        # Split the group into parallel planes (subclusters)
        subclusters_local, noise_idx_local, s_vals = _split_by_offset_along_normal(
            group_pts, proj_normal, plane_distance_eps, min_points_per_plane
        )

        # Handle points considered noise by the 1D split
        if noise_strategy_layer == "new_plane":
            new_from_noise = _make_new_plane_from_noise_if_large(noise_idx_local, min_points_per_plane)
            if new_from_noise:
                subclusters_local.extend(new_from_noise)
                noise_idx_local = np.array([], dtype=int)
            subclusters_local = _assign_noise_points_to_layers(
                s_vals, noise_idx_local, subclusters_local, plane_distance_eps, "merge"
            )
        elif noise_strategy_layer in ("force_assign", "merge"):
            subclusters_local = _assign_noise_points_to_layers(
                s_vals, noise_idx_local, subclusters_local, plane_distance_eps, noise_strategy_layer
            )
        
        # Refine and store each final plane
        for sub_idx_local in subclusters_local:
            sub_pts = group_pts[sub_idx_local]
            
            if refinement_method.lower() == "ransac" and PYRANSAC_AVAILABLE:
                fitter = pyrsc.Plane()
                plane_eq, inlier_local = fitter.fit(sub_pts, thresh=plane_distance_eps, maxIteration=ransac_max_iterations)
                n = plane_eq[:3]
                plane_eq = np.array([*(n / (np.linalg.norm(n) + 1e-12)), plane_eq[3]], dtype=np.float64)
            else: # Default to PCA
                plane_eq = _fit_plane_with_pca(sub_pts)
                inlier_local = np.arange(len(sub_pts))

            if len(inlier_local) < min_points_per_plane:
                continue

            # Map local indices back to global indices
            final_in_subgroup_idx = sub_idx_local[inlier_local]
            final_in_group_idx = group_idx_local[final_in_subgroup_idx]
            global_inliers = global_indices_of_subset[final_in_group_idx]

            found_planes.append((plane_eq, global_inliers))

    # 5) FINAL STEP: Optionally assign unassigned points to the closest found plane
    if assign_unassigned_to_closest_plane and found_planes:
        unassigned_idx_local = np.where(labels_orient == -1)[0]
        
        if unassigned_idx_local.size > 0:
            unassigned_pts = points_subset[unassigned_idx_local]
            num_unassigned = len(unassigned_pts)
            num_planes = len(found_planes)
            
            plane_equations = np.array([p[0] for p in found_planes])
            
            # Calculate distance from each unassigned point to each plane
            dists = np.abs(unassigned_pts @ plane_equations[:, :3].T + plane_equations[:, 3])
            
            # Find the closest plane and its distance for each point
            closest_plane_indices = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(num_unassigned), closest_plane_indices]
            
            # Filter points that are within the threshold of their closest plane
            assignment_mask = min_dists <= plane_distance_eps
            
            points_to_assign_indices = unassigned_idx_local[assignment_mask]
            planes_to_receive_indices = closest_plane_indices[assignment_mask]
            
            # Efficiently group points to be added to each plane
            additions = [[] for _ in range(num_planes)]
            for i in range(len(points_to_assign_indices)):
                local_idx = points_to_assign_indices[i]
                plane_idx = planes_to_receive_indices[i]
                global_idx = global_indices_of_subset[local_idx]
                additions[plane_idx].append(global_idx)
            
            # Update the inlier lists for the planes with the new points
            num_reassigned = 0
            for i, plane_additions in enumerate(additions):
                if plane_additions:
                    plane_eq, old_inliers = found_planes[i]
                    new_inliers = np.concatenate((old_inliers, np.array(plane_additions, dtype=old_inliers.dtype)))
                    found_planes[i] = (plane_eq, new_inliers)
                    num_reassigned += len(plane_additions)
            
            print(f"     [INFO] Reassigned {num_reassigned} / {num_unassigned} unassigned points to nearest planes.")
    
    return found_planes



# Go through component ids and run RANSAC on each one
def extract_primitives_by_component(
    vignette: 'ProcessedVignette',
    primitive_type: str, # plane, cylinder, sphere, or cuboid
    distance_threshold: float, # RANSAC distance threshold
    min_points: int, # min number of inliers
    plane_method: str = 'region_growing', # 'region_growing' or 'ransac'
    # For region growing planes
    normal_angle_tolerance_deg: float = 15.0, 
    min_samples_normals_ratio: float = 0.1,
    noise_strategy_layer: str = "new_plane" 
) -> None:
    """
    Extracts primitives by grouping points based on a shared attribute value.

    For each unique value in the specified attribute (e.g., for each component_id),
    this function runs RANSAC to find primitives and then composes the results.
    """
    plural_name = f"{primitive_type}s"
    id_name = f"{primitive_type}_id"
    base_attribute_name = "component_id"

    print(f"Extracting {plural_name} grouped by '{base_attribute_name}'...")
    if primitive_type == 'plane':
        print(f"   - Using plane fitting method: '{plane_method}'")
    vignette.clear_abstractions(plural_name, auto_save=False)
    
    # --- Setup for collecting results ---
    all_found_abstractions = []
    final_labels = np.zeros(vignette.n_points, dtype=int)
    current_primitive_id = 1
    
    groups_to_process = []
    temp_attr_name = None

    # --- Determine which groups to process ---
    labels = vignette.get_attribute(base_attribute_name)
    unique_values = np.unique(labels) if labels is not None else []
    num_components = len(unique_values) - (1 if -1 in unique_values else 0)

    if labels is None or num_components <= 1:
        print(f"   - No components found or only one component. Falling back to entire point cloud.")
        # Create a temporary attribute that groups all points into a single component (value=0)
        temp_attr_name = "__temp_group_id"
        vignette.set_attribute(temp_attr_name, np.zeros(vignette.n_points, dtype=int), auto_save=False)
        groups_to_process.append({'attribute_name': temp_attr_name, 'value': 0, 'display_name': 'global'})
    else:
        print(f"   - Found {num_components} components. Processing each separately.")
        for value in unique_values:
            if value == -1: continue # Skip noise
            groups_to_process.append({'attribute_name': base_attribute_name, 'value': value, 'display_name': f'component_{value}'})
    
    try:
        # --- Main Loop: Iterate through the prepared groups ---
        for group in groups_to_process:
            attr_name, attr_value = group['attribute_name'], group['value']
            print(f"   - Processing subset where '{attr_name}' == {attr_value}...")
            
            found_primitives = []
            if primitive_type == 'plane' and plane_method == 'region_growing':
                # For planes with region growing, we need the points and global indices
                group_indices = np.where(vignette.get_attribute(attr_name) == attr_value)[0]
                if len(group_indices) > 0:
                    group_points = vignette.points[group_indices]
                    found_primitives = _region_growing_plane_segmentation(
                        points_subset=group_points,
                        global_indices_of_subset=group_indices,
                        normal_search_radius=distance_threshold * 5,
                        normal_max_nn=30,
                        normal_angle_tolerance_deg=normal_angle_tolerance_deg,
                        min_samples_normals_ratio=min_samples_normals_ratio,
                        plane_distance_eps=distance_threshold,
                        min_points_per_plane=min_points,
                        refinement_method="pca",
                        ransac_max_iterations=500,
                        noise_strategy_layer=noise_strategy_layer
                    )
            else:
                # For all other cases (cylinders, spheres, or planes with 'ransac' method)
                found_primitives = _fit_primitive_to_subset(vignette, primitive_type, attr_name, attr_value, distance_threshold, min_points)
            
            if not found_primitives:
                print("     - No primitives found for this subset.")
                continue
            
            # (The rest of the logic is the same as your correct implementation)
            print(f"     - Found {len(found_primitives)} primitive(s).")
            for params, global_inlier_indices in found_primitives:
                inlier_points = vignette.points[global_inlier_indices]
                abstraction_data, fit_error = {}, 0.0
                
                if primitive_type == 'plane':
                    equation = np.array(params); dists = np.abs(inlier_points @ equation[:3] + equation[3]) / np.linalg.norm(equation[:3])
                    fit_error = np.mean(dists); abstraction_data.update({"equation": list(params)})
                
                
                elif primitive_type == 'cylinder':
                    center, axis, radius = np.array(params[0]), np.array(params[1]), params[2]; vecs = inlier_points - center
                    dists_to_axis = np.linalg.norm(np.cross(vecs, axis), axis=1) / np.linalg.norm(axis)
                    fit_error = np.mean(np.abs(dists_to_axis - radius)); abstraction_data.update({"center": center.tolist(), "axis": axis.tolist(), "radius": radius})
                elif primitive_type == 'sphere':
                    center, radius = np.array(params[0]), np.array(params[1]); dists_to_center = np.linalg.norm(inlier_points - center, axis=1)
                    fit_error = np.mean(np.abs(dists_to_center - radius)); abstraction_data.update({"center": center.tolist(), "radius": radius})
                
                if len(inlier_points) > 3:
                    obb_pcd = o3d.geometry.PointCloud(); obb_pcd.points = o3d.utility.Vector3dVector(inlier_points)
                    obb = obb_pcd.get_oriented_bounding_box()
                    abstraction_data.update({"obb_center": obb.center.tolist(), "obb_rotation": obb.R.tolist(), "obb_extent": obb.extent.tolist()})
                    if primitive_type == 'cuboid':
                        points_local = (inlier_points - obb.center) @ obb.R.T; half_extents = obb.extent / 2.0
                        dists_from_surface = np.maximum(0, np.abs(points_local) - half_extents); point_errors = np.linalg.norm(dists_from_surface, axis=1)
                        fit_error = np.mean(point_errors)
                
                final_labels[global_inlier_indices] = current_primitive_id
                
                abstraction_data.update({
                    id_name: current_primitive_id,
                    "source_attribute": {base_attribute_name: group['display_name']},
                    "point_count": len(global_inlier_indices), "fit_error": fit_error,
                    "point_indices": global_inlier_indices.tolist()
                })
                
                all_found_abstractions.append(abstraction_data)
                current_primitive_id += 1
    
    finally:
        # --- Clean up the temporary attribute if it was created ---
        if temp_attr_name and hasattr(vignette, temp_attr_name):
            del vignette.attributes[temp_attr_name]
            print(f"   - Cleaned up temporary attribute '{temp_attr_name}'.")

    if all_found_abstractions:
        vignette.metadata.setdefault('abstractions', {})[plural_name] = all_found_abstractions
        print(f"\n   - Added a total of {len(all_found_abstractions)} new abstractions of type '{plural_name}'.")

    vignette.set_attribute(id_name, final_labels, auto_save=True)
    print(f"Finished extracting dominant {plural_name}.")


# Best Fit
def compose_best_fit_abstraction(
    vignette: 'ProcessedVignette', 
    min_coverage_ratio: float = 0.01,
    score_alpha: float = 1.0,
    default_distance_threshold: float = 0.02
) -> None:
    """
    Analyzes all found primitives and composes a "best fit" scene using a
    tunable scoring system that balances point coverage with fit quality.

    Args:
        vignette: The vignette to process.
        min_coverage_ratio: The minimum percentage of *new* points a primitive must
                            explain to be included (e.g., 0.01 = 1%).
        score_alpha: An exponent to control the penalty for fit_error.
                     - alpha > 1: Prioritizes precision (low error).
                     - alpha < 1: Prioritizes coverage (high point count).
                     - alpha = 1: Balanced.
        default_distance_threshold: A fallback error value for primitives
                                    where fit_error wasn't calculated.
    """
    print("Composing best-fit abstraction...")
    print(f"   - Parameters: min_coverage_ratio={min_coverage_ratio}, score_alpha={score_alpha}")
    vignette.clear_abstractions('best_fit_composition', auto_save=False)

    all_primitives = []
    candidate_abstractions = vignette.get_abstractions() or {}
    
    for primitive_type, primitives in candidate_abstractions.items():
        if primitive_type not in ['planes', 'cylinders', 'spheres', 'cuboids']:
            continue
        for primitive in primitives:
            primitive['type'] = primitive_type
            
            # --- Tunable, Balanced Scoring System ---
            point_count = primitive.get('point_count', 0)
            fit_error = primitive.get('fit_error', default_distance_threshold)
            epsilon = 1e-6
            
            # --- MODIFIED: Implemented the new weighted score formula ---
            primitive['score'] = point_count / ((fit_error + epsilon) ** score_alpha)
            all_primitives.append(primitive)
            
    if not all_primitives:
        print("   - No candidate primitives found to compose. Aborting.")
        return
            
    all_primitives.sort(key=lambda p: p['score'], reverse=True)
    
    unexplained_points_mask = np.ones(vignette.n_points, dtype=bool)
    final_composition = []
    best_fit_labels = np.full(vignette.n_points, -1, dtype=int) # Use -1 for unassigned
    current_best_fit_id = 1
    
    min_points_to_consider = int(vignette.n_points * min_coverage_ratio)

    print(f"   - Starting greedy selection. A primitive must explain at least {min_points_to_consider} new points.")

    for i, primitive in enumerate(all_primitives):
        point_indices = np.array(primitive.get('point_indices', []))
        if len(point_indices) == 0:
            continue

        unexplained_inliers_mask = unexplained_points_mask[point_indices]
        num_newly_explained = np.sum(unexplained_inliers_mask)
        
        primitive_id_key = f"{primitive['type'][:-1]}_id" # e.g., 'plane_id'
        primitive_id = primitive.get(primitive_id_key, 'N/A')

        print(f"\n     + Evaluating Candidate #{i+1}: {primitive['type']} (Source ID: {primitive_id})")
        print(f"       - Stats: Point Count={primitive.get('point_count', 0)}, Fit Error={primitive.get('fit_error', 'N/A'):.4f}")
        print(f"       - Calculated Score: {primitive['score']:.2f}")
        print(f"       - It has {len(point_indices)} total points. Of those, {num_newly_explained} are currently unexplained.")

        if num_newly_explained >= min_points_to_consider:
            print(f"       - ACCEPTED: {num_newly_explained} >= threshold {min_points_to_consider}.")
            
            primitive['best_fit_id'] = current_best_fit_id
            final_composition.append(primitive)
            
            newly_explained_indices = point_indices[unexplained_inliers_mask]
            best_fit_labels[newly_explained_indices] = current_best_fit_id
            
            unexplained_points_mask[newly_explained_indices] = False
            current_best_fit_id += 1
        else:
            print(f"       - REJECTED: {num_newly_explained} < threshold {min_points_to_consider}.")
    
    num_explained = vignette.n_points - np.sum(unexplained_points_mask)
    coverage_percent = (num_explained / vignette.n_points) * 100 if vignette.n_points > 0 else 0
    
    print("\nFinished composing best-fit abstraction.")
    print(f"   - Selected {len(final_composition)} primitives for the final composition.")
    print(f"   - Total points explained: {num_explained} / {vignette.n_points} ({coverage_percent:.2f}%)")
    
    vignette.metadata['best_fit_composition'] = final_composition
    vignette.set_attribute('best_fit_id', best_fit_labels, auto_save=True)


# Symmetry

def _reflect_points(points, plane_normal, plane_point):
    normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
    d = -np.dot(normal, np.array(plane_point))
    dists = np.dot(points, normal) + d
    return points - 2 * dists[:, np.newaxis] * normal


def _score_reflection_plane_robust(points, point_tree, plane_normal, plane_point, match_threshold):
    """Robustly scores a reflection plane using bidirectional consistency."""
    # Forward check
    reflected_points = _reflect_points(points, plane_normal, plane_point)
    distances, neighbor_indices = point_tree.query(reflected_points, k=1)
    
    potential_inlier_mask = distances < match_threshold
    potential_indices = np.where(potential_inlier_mask)[0]
    
    # --- [DEBUG] Print results of the initial forward check ---
    print(f"      [DEBUG] Potential inliers found (distance < {match_threshold}m): {len(potential_indices)} / {len(points)}")

    # Backward consistency check
    consistent_inlier_indices = []
    if len(potential_indices) > 0:
        neighbors_to_check = points[neighbor_indices[potential_indices]]
        reflected_neighbors = _reflect_points(neighbors_to_check, plane_normal, plane_point)
        _, reverse_neighbor_indices = point_tree.query(reflected_neighbors, k=1)
        
        is_consistent = (reverse_neighbor_indices == potential_indices)
        consistent_inlier_indices = potential_indices[is_consistent]
        # --- [DEBUG] Print results of the backward consistency check ---
        print(f"      [DEBUG] Consistent inliers found (passed backward check): {len(consistent_inlier_indices)} / {len(potential_indices)}")

    if len(consistent_inlier_indices) == 0:
        return {'score': float('inf'), 'inlier_ratio': 0, 'mean_error': float('inf')}
        
    inlier_ratio = len(consistent_inlier_indices) / len(points)
    mean_error = np.mean(distances[consistent_inlier_indices])
    score = mean_error / inlier_ratio if inlier_ratio > 0 else float('inf')
    
    return {'score': score, 'inlier_ratio': inlier_ratio, 'mean_error': mean_error}


def analyze_global_symmetry(
    vignette: 'ProcessedVignette', 
    match_threshold: float = 0.02, 
    min_inlier_ratio: float = 0.5,
    auto_save: bool = False
) -> None:
    print("Analyzing global symmetry (robust method)...")
    vignette.clear_abstractions('symmetries', auto_save=False)

    all_props = vignette.get_abstractions('structural_properties')
    if not all_props:
        print("   [DEBUG] EXIT: Structural properties not found. Run PCA first.")
        return
    global_props = next((p for p in all_props if p['type'] == 'global'), None)
    if not global_props:
        print("   [DEBUG] EXIT: Global structural properties not found.")
        return

    points = vignette.points
    component_labels = vignette.get_attribute('component_id')
    print(f"   [DEBUG] Initial point count: {len(points)}")
    if component_labels is not None:
        print("   [DEBUG] 'component_id' attribute found. Attempting to filter noise...")
        non_noise_mask = component_labels != -1
        num_non_noise = np.sum(non_noise_mask)
        
        # Check if filtering would leave enough points
        if num_non_noise >= 100:
            points_to_analyze = points_to_analyze[non_noise_mask]
            print(f"   [DEBUG] Successfully filtered to {len(points_to_analyze)} non-noise points.")
        else:
            print(f"   [DEBUG] WARNING: Found only {num_non_noise} non-noise points, which is below the threshold.")
            print(f"   [DEBUG] This likely means component segmentation failed. Using ALL points as a fallback.")

    if len(points) < 100:
        print(f"   [DEBUG] EXIT: Not enough points for reliable analysis (found {len(points)}, need 100).")
        return

    point_tree = KDTree(points)
    centroid = np.array(global_props['centroid'])
    axes = np.array(global_props['axes'])

    best_reflection = {'score': float('inf')}
    axis_names = ['Primary', 'Secondary', 'Tertiary']

    print("   --- Testing Reflectional Symmetry ---")
    for i in range(3):
        plane_normal = axes[i]
        print(f"   -> Testing plane normal to '{axis_names[i]}' axis...")
        result = _score_reflection_plane_robust(points, point_tree, plane_normal, centroid, match_threshold)
        print(f"      [DEBUG] Result: Score={result['score']:.4f}, Inlier Ratio={result['inlier_ratio']:.2%}, Mean Error={result['mean_error']:.4f}m")
        
        if result['score'] < best_reflection['score']:
            best_reflection = result
            best_reflection['plane_normal'] = plane_normal.tolist()

    print("   --- Final Decision ---")
    best_inlier_ratio = best_reflection.get('inlier_ratio', 0)
    print(f"   [DEBUG] Best inlier ratio found: {best_inlier_ratio:.2%}. Required minimum: {min_inlier_ratio:.2%}")
    
    if best_inlier_ratio > min_inlier_ratio:
        print(f"   SUCCESS: Found reflectional symmetry that meets the criteria!")
        reflection_result = {
            'type': 'reflectional',
            'score': best_reflection['score'],
            'inlier_ratio': best_reflection['inlier_ratio'],
            'mean_inlier_error': best_reflection['mean_error'],
            'plane_point': centroid.tolist(),
            'plane_normal': best_reflection['plane_normal']
        }
        vignette.add_abstraction('symmetries', reflection_result, auto_save=False)
    else:
        print("   - No reflectional symmetry found that passed the minimum inlier ratio threshold.")

    # (Rotational symmetry part would go here)
    
    print("Finished symmetry analysis.")
    if auto_save and vignette.file_path:
        vignette.save()


# --- 2.4 Structural Analysis ---

# Inter primitive relations

def analyze_primitive_relations(
    vignette: 'ProcessedVignette',
    angle_tolerance_deg: float = 5.0,
    distance_tolerance_m: float = 0.01,
    auto_save: bool = False
) -> None:
    """
    Analyzes and stores geometric relationships between primitives in a vignette.
    
    This version supports:
    - Plane-Plane: Parallelism, Perpendicularity, Co-planarity, Distance.
    - Cylinder-Cylinder: Parallelism, Perpendicularity.
    
    (Includes detailed debug logging).
    """
    print("Analyzing primitive relationships...")
    vignette.clear_abstractions('primitive_relations', auto_save=False)
    
    # --- Calculate tolerance values from degrees ---
    # cos(angle) for parallel check. Angle is close to 0 or 180.
    parallel_dot_threshold = np.cos(np.deg2rad(angle_tolerance_deg))
    # cos(angle) for perpendicular check. Angle is close to 90.
    perp_dot_threshold = np.cos(np.deg2rad(90 - angle_tolerance_deg))
    
    print(f"   [DEBUG] Angle tolerance: {angle_tolerance_deg}°")
    print(f"   [DEBUG] Parallel check: dot product must be > {parallel_dot_threshold:.4f}")
    print(f"   [DEBUG] Perpendicular check: dot product must be < {perp_dot_threshold:.4f}")

    # --- Retrieve all primitives at the start ---
    planes = vignette.get_abstractions('planes') or []
    cylinders = vignette.get_abstractions('cylinders') or []

    # --- 1. Analyze Plane-Plane Relationships ---
    if len(planes) < 2:
        print("\n   - Skipping plane-plane analysis (less than 2 planes found).")
    else:
        print(f"\n   --- Analyzing {len(planes)} Plane-Plane pairs ---")
        for plane1, plane2 in itertools.combinations(planes, 2):
            id1 = plane1['plane_id']
            id2 = plane2['plane_id']
            
            print(f"\n   -> Comparing plane_{id1} and plane_{id2}...")

            # Extract and normalize normal vectors
            n1 = np.array(plane1['equation'][:3]); n1 /= np.linalg.norm(n1)
            n2 = np.array(plane2['equation'][:3]); n2 /= np.linalg.norm(n2)
            dot_product = np.abs(np.dot(n1, n2))
            
            print(f"      [DEBUG] Normalized dot product: {dot_product:.4f}")

            # Check for Parallelism
            if dot_product > parallel_dot_threshold:
                print(f"      [DEBUG] Test PASS: {dot_product:.4f} > {parallel_dot_threshold:.4f} (Parallel)")
                
                # Find a point on plane 1 to calculate distance
                eq1 = plane1['equation']
                if abs(eq1[2]) > 1e-6: p1 = np.array([0, 0, -eq1[3]/eq1[2]])
                elif abs(eq1[1]) > 1e-6: p1 = np.array([0, -eq1[3]/eq1[1], 0])
                else: p1 = np.array([-eq1[3]/eq1[0], 0, 0])
                
                eq2 = plane2['equation']
                distance = np.abs(np.dot(p1, eq2[:3]) + eq2[3]) / np.linalg.norm(eq2[:3])
                print(f"      [DEBUG] Distance between planes: {distance:.4f}m")

                relation_type = "co-planar" if distance < distance_tolerance_m else "parallel"
                print(f"      -> SUCCESS: Found '{relation_type.upper()}' relationship.")
                
                relation = {
                    'type': relation_type,
                    'primitives': [f'plane_{id1}', f'plane_{id2}'],
                    'angle_diff_deg': np.rad2deg(np.arccos(dot_product)),
                    'distance_m': distance if relation_type == 'parallel' else 0.0
                }
                vignette.add_abstraction('primitive_relations', relation, auto_save=False)

            # Check for Perpendicularity
            elif dot_product < perp_dot_threshold:
                print(f"      [DEBUG] Test PASS: {dot_product:.4f} < {perp_dot_threshold:.4f} (Perpendicular)")
                print(f"      -> SUCCESS: Found 'PERPENDICULAR' relationship.")
                relation = {
                    'type': 'perpendicular',
                    'primitives': [f'plane_{id1}', f'plane_{id2}'],
                    'angle_diff_deg': 90.0 - np.rad2deg(np.arcsin(dot_product))
                }
                vignette.add_abstraction('primitive_relations', relation, auto_save=False)
            
            else:
                 print(f"      [DEBUG] Test FAIL: No significant relationship found.")

    # --- 2. Analyze Cylinder-Cylinder Relationships ---
    if len(cylinders) < 2:
        print("\n   - Skipping cylinder-cylinder analysis (less than 2 cylinders found).")
    else:
        print(f"\n   --- Analyzing {len(cylinders)} Cylinder-Cylinder pairs ---")
        for cyl1, cyl2 in itertools.combinations(cylinders, 2):
            id1 = cyl1['cylinder_id']
            id2 = cyl2['cylinder_id']
            print(f"\n   -> Comparing cylinder_{id1} and cylinder_{id2}...")
            
            a1 = np.array(cyl1['axis']); a1 /= np.linalg.norm(a1)
            a2 = np.array(cyl2['axis']); a2 /= np.linalg.norm(a2)
            dot_product = np.abs(np.dot(a1, a2))
            print(f"      [DEBUG] Normalized dot product: {dot_product:.4f}")
            
            if dot_product > parallel_dot_threshold:
                print(f"      -> SUCCESS: Found 'PARALLEL' relationship.")
                relation = {'type': 'parallel', 'primitives': [f'cylinder_{id1}', f'cylinder_{id2}']}
                vignette.add_abstraction('primitive_relations', relation, auto_save=False)
            elif dot_product < perp_dot_threshold:
                print(f"      -> SUCCESS: Found 'PERPENDICULAR' relationship.")
                relation = {'type': 'perpendicular', 'primitives': [f'cylinder_{id1}', f'cylinder_{id2}']}
                vignette.add_abstraction('primitive_relations', relation, auto_save=False)
            else:
                print(f"      [DEBUG] Test FAIL: No significant relationship found.")
                
    if auto_save and vignette.file_path:
        vignette.save()
    print("\nFinished analyzing primitive relationships.")













# Older versions



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
                
                # The finalhj=gv fit error is the average of these distances.
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



# Older code

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

