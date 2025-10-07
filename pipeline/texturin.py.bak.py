"""
texturing.py - Functions for creating textured meshes from abstracted primitives.

This module takes a vignette that has already been analyzed by the abstraction
pipeline and projects the original RGB image onto the surfaces of the detected
primitives, creating a set of textured 3D "scraps" similar to Mental Canvas.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import shutil # Import the shutil module for efficient file copying
from PIL import Image, ImageDraw # Import ImageDraw to create masks

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
    # Project points onto the image plane
    projected = intrinsics @ points_3d.T
    
    # Get the depth value (d) for each point
    d = projected[2, :] + 1e-6 # Add epsilon to avoid division by zero
    
    # Normalize to get final pixel coordinates (u, v)
    u = projected[0, :] / d
    v = projected[1, :] / d
    
    return np.vstack((u, v)).T


def create_textured_planes(
    vignette: ProcessedVignette,
    rgb_image_path: str,
    output_dir: str,
    point_radius: int = 5
) -> None:
    """
    Generates uniquely textured and masked meshes for all dominant planes.

    This implementation creates a distinct, non-rectangular texture for each
    plane, achieving a "Mental Canvas" style effect.

    For each plane, this function:
    1. Projects all of the plane's 3D points into 2D to find their footprint.
    2. Creates an alpha mask by drawing circles at each projected point's location.
    3. Crops the source RGB image to the minimal bounding box of the footprint.
    4. Applies the alpha mask to the cropped texture, making other areas transparent.
    5. Saves this new unique texture as a PNG asset.
    6. Calculates the correct UV coordinates for the 3D mesh to map to this new texture.
    7. Stores the mesh geometry, UVs, and path to the unique texture in the metadata.

    Args:
        vignette: The vignette, which must have 'planes' in its metadata.
        rgb_image_path: The file path to the original, full-resolution RGB image.
        output_dir: The directory where the vignette and its assets are stored.
        point_radius: The radius of the circle to draw around each point for the mask.
    """
    print("--- Starting Textured Plane Creation ---")
    vignette.clear_abstractions('textured_planes', auto_save=False)
    
    # === Step 1: Load Prerequisites ===
    print("\n[Step 1] Loading prerequisites...")
    plane_abstractions = vignette.get_abstractions('planes')
    if not plane_abstractions:
        print("   - No planes found in vignette to texture. Aborting.")
        return

    # Load camera data needed for projection
    capture_meta = vignette.metadata.get('capture_metadata', {})
    intrinsics = np.array(capture_meta.get('camera_intrinsics', {}).get('columns', np.identity(3))).T
    center_offset = np.array(capture_meta.get('center_offset', [0,0,0]))
    
    try:
        source_image = Image.open(rgb_image_path).convert("RGBA")
        original_image_size = source_image.size
    except Exception as e:
        print(f"   - ERROR: Could not read source image at {rgb_image_path}. Error: {e}")
        return

    if np.array_equal(intrinsics, np.identity(3)) or np.allclose(center_offset, [0,0,0]):
         print("   - ERROR: Camera intrinsics or center_offset not found in vignette metadata.")
         return
    print(f"   - Prerequisites loaded successfully.")

    # === Step 2: Prepare Asset Folder ===
    print("\n[Step 2] Preparing asset folder...")
    assets_path = Path(output_dir) / "assets" / "planes"
    assets_path.mkdir(parents=True, exist_ok=True)
    print(f"   - Assets will be saved in: {assets_path}")
    
    textured_planes_list = []

    for plane_info in plane_abstractions:
        plane_id = plane_info['plane_id']
        print(f"\n--- Processing Plane ID: {plane_id} ---")
        if "obb_center" not in plane_info:
            print("   - Skipping plane, missing OBB data.")
            continue

        # === Step 3: Project all inlier points to get footprint ===
        print("[Step 3] Projecting all inlier points to 2D...")
        inlier_points_3d_centered = vignette.points[plane_info['point_indices']]
        inlier_points_3d_original = inlier_points_3d_centered + center_offset
        inlier_points_2d = project_points_to_2d(inlier_points_3d_original, intrinsics)

        # === Step 4: Create Full-Size Alpha Mask ===
        print("[Step 4] Creating full-size alpha mask...")
        # Create a blank, transparent image that is the same size as the original.
        final_texture = Image.new('RGBA', original_image_size, (0, 0, 0, 0))

        # Create a separate mask image (grayscale 'L' mode is efficient).
        mask = Image.new('L', original_image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        # The projected points are already in the correct coordinate system for the full-size mask.
        for p in inlier_points_2d:
            draw.ellipse((p[0]-point_radius, p[1]-point_radius, p[0]+point_radius, p[1]+point_radius), fill=255)
        
        # Composite the original image onto our blank canvas, using the mask to "punch out" the shape.
        final_texture.paste(source_image, (0, 0), mask)
        print(f"   - Created masked texture of size {final_texture.size}")

        # === Step 5: Save Unique Texture Asset ===
        print("[Step 5] Saving unique texture asset...")
        texture_filename = f"plane_{plane_id}_texture.png"
        texture_filepath = assets_path / texture_filename
        final_texture.save(texture_filepath)
        print(f"   - Saved asset to: {texture_filepath}")
        
        # === Step 6: Define 3D Mesh Vertices ===
        print("[Step 6] Defining 3D mesh vertices from OBB...")
        center, rotation, extent = np.array(plane_info['obb_center']), np.array(plane_info['obb_rotation']), np.array(plane_info['obb_extent'])
        axes, sorted_indices = [rotation[:, i] for i in range(3)], np.argsort(extent)
        major_axis, minor_axis = axes[sorted_indices[2]], axes[sorted_indices[1]]
        major_half, minor_half = extent[sorted_indices[2]]/2.0, extent[sorted_indices[1]]/2.0
        v1, v2, v3, v4 = center-major_axis*major_half+minor_axis*minor_half, center+major_axis*major_half+minor_axis*minor_half, center+major_axis*major_half-minor_axis*minor_half, center-major_axis*major_half-minor_axis*minor_half
        corners_3d_centered = np.array([v1, v2, v3, v4])
        mesh_vertices, mesh_faces = [v.tolist() for v in corners_3d_centered], [[0, 1, 3], [1, 2, 3]]
        
        # === Step 7: Calculate UVs Relative to Full Image ===
        print("[Step 7] Calculating UV coordinates for the full image...")
        corners_3d_original = corners_3d_centered + center_offset
        corners_2d = project_points_to_2d(corners_3d_original, intrinsics)
        # Normalize by the size of the *full original image*.
        uv_corners = corners_2d / np.array(original_image_size)
        print(f"   - Calculated UV coordinates for 4 corners:\n{uv_corners.round(3)}")
        
        # === Step 8: Store Final Data in Vignette ===
        print("[Step 8] Storing textured plane data in vignette metadata...")
        textured_plane_data = {
            "plane_id": plane_id,
            "mesh_vertices": mesh_vertices,
            "mesh_faces": mesh_faces,
            "mesh_uvs": uv_corners.tolist(),
            "texture_path": str(Path("assets") / "planes" / texture_filename)
        }
        textured_planes_list.append(textured_plane_data)

    vignette.metadata['textured_abstractions'] = {'planes': textured_planes_list}
    vignette.save()
    print("\n--- Finished Advanced Textured Plane Creation ---")




def _project_point_to_plane(p_world: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Orthogonally project a 3D point onto a plane defined by (plane_point, plane_normal).

    Args:
        p_world: (3,) original 3D point (camera/world coords, same frame as plane_point).
        plane_point: (3,) any point on the plane (e.g., the plane OBB center in original coords).
        plane_normal: (3,) unit-length plane normal in world/camera frame.

    Returns:
        (3,) the closest point to p_world on the plane.
    """
    v = p_world - plane_point
    dist = np.dot(v, plane_normal)
    return p_world - dist * plane_normal


def _sample_disk_color_rgba(img: Image.Image, uv: np.ndarray, radius: int) -> tuple:
    """
    Sample the average RGBA color inside a disk centered at pixel 'uv' with 'radius'.

    Notes:
        - Clamps to image bounds.
        - Returns a 4-tuple (R, G, B, A), A is 255 if there are samples, otherwise 0.

    Args:
        img: PIL RGBA image.
        uv: (2,) array (u, v) pixel coordinates (float allowed).
        radius: sample radius in pixels.

    Returns:
        tuple(int,int,int,int): averaged RGBA.
    """
    w, h = img.size
    cx, cy = float(uv[0]), float(uv[1])
    x_min = max(int(np.floor(cx - radius)), 0)
    x_max = min(int(np.ceil (cx + radius)), w - 1)
    y_min = max(int(np.floor(cy - radius)), 0)
    y_max = min(int(np.ceil (cy + radius)), h - 1)

    if x_min > x_max or y_min > y_max:
        return (0, 0, 0, 0)

    # Fast path: crop to bounding box, then mask a circular area
    crop = img.crop((x_min, y_min, x_max + 1, y_max + 1)).convert("RGBA")
    cw, ch = crop.size

    # Build a circular alpha mask the same size as the crop
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    # Circle center in crop coords:
    ccx, ccy = cx - x_min, cy - y_min
    draw.ellipse((ccx - radius, ccy - radius, ccx + radius, ccy + radius), fill=255)

    # Apply mask, then compute mean over nonzero alpha
    crop_np = np.asarray(crop, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.uint8)

    sel = mask_np > 0
    if not np.any(sel):
        return (0, 0, 0, 0)

    # Compute per-channel average over selected pixels
    sel3 = np.repeat(sel[..., None], 4, axis=2)
    vals = crop_np[sel3].reshape(-1, 4)
    mean = vals.mean(axis=0)

    r, g, b, a = [int(np.clip(v, 0, 255)) for v in mean]
    # Force alpha to fully opaque for “painted” dots (you can change if you want soft alpha)
    return (r, g, b, 255)


def create_textured_planes_revised(
    vignette: ProcessedVignette,
    rgb_image_path: str,
    output_dir: str,
    point_draw_radius: int = 10,
    color_sample_radius: int = 10
) -> None:
    """
    Revised texture creation with 3-step dot logic:

    For each inlier *point* on a plane:
      1) Project the original 3D point to the RGB image (camera plane) and
         sample the average color inside a disk (color_sample_radius).
      2) Project that 3D point orthogonally to its *own* plane (closest point).
      3) Reproject the plane-projected 3D point to the RGB image; this is where
         the colored dot should be *drawn*. Paint a filled circle (point_draw_radius)
         at this location using the sampled color from step 1.

    Intuition:
      - The *color* comes from where the camera actually saw the 3D point (step 1).
      - The *placement* of that color on the texture is where the point “belongs” on
        the planar surface (step 2), viewed again through the camera (step 3).
      - This creates a texture that “collects” colors by nearest planar assignment,
        rather than directly stamping points where they were first seen.

    Other behavior (assets folder, per-plane PNGs, per-plane quad UVs) matches your original.

    Args:
        vignette: ProcessedVignette with 'planes' abstractions and capture metadata.
        rgb_image_path: Path to the source RGB image used for coloring.
        output_dir: Base folder for saving per-plane texture assets.
        point_draw_radius: Radius (px) of the drawn disk for each projected dot.
        color_sample_radius: Radius (px) of the *sampling* disk for color averaging.
    """
    print("--- Starting Revised Textured Plane Creation ---")
    vignette.clear_abstractions('textured_planes', auto_save=False)

    # === Load plane abstractions and camera data ===
    plane_abstractions = vignette.get_abstractions('planes')
    if not plane_abstractions:
        print("   - No planes found in vignette. Aborting.")
        return

    capture_meta = vignette.metadata.get('capture_metadata', {})
    intrinsics = np.array(capture_meta.get('camera_intrinsics', {}).get('columns', np.identity(3))).T
    center_offset = np.array(capture_meta.get('center_offset', [0, 0, 0]))

    try:
        source_image = Image.open(rgb_image_path).convert("RGBA")
        original_image_size = source_image.size
    except Exception as e:
        print(f"   - ERROR: Could not read source image at {rgb_image_path}. Error: {e}")
        return

    if np.array_equal(intrinsics, np.identity(3)) or np.allclose(center_offset, [0, 0, 0]):
        print("   - ERROR: Camera intrinsics or center_offset not found in vignette metadata.")
        return

    # === Prepare output folder ===
    assets_path = Path(output_dir) / "assets" / "planes"
    assets_path.mkdir(parents=True, exist_ok=True)

    textured_planes_list = []

    # === Process each plane ===
    for plane_info in plane_abstractions:
        plane_id = plane_info.get('plane_id')
        print(f"\n--- Processing Plane ID: {plane_id} ---")
        if "obb_center" not in plane_info or "obb_rotation" not in plane_info or "obb_extent" not in plane_info:
            print("   - Skipping plane, missing OBB data.")
            continue

        # 3D points: centered coords in vignette; convert to original (camera) coords
        inlier_points_3d_centered = vignette.points[plane_info['point_indices']]
        inlier_points_3d_original = inlier_points_3d_centered + center_offset

        # Determine plane geometry in original coords:
        center_c = np.array(plane_info['obb_center']) + center_offset  # plane point in original/camera frame
        R = np.array(plane_info['obb_rotation'])  # columns are local axes
        extent = np.array(plane_info['obb_extent'])

        # The *smallest* OBB extent corresponds to plane thickness → its axis is the plane normal.
        axes = [R[:, i] for i in range(3)]
        sorted_indices = np.argsort(extent)
        plane_normal = axes[sorted_indices[0]]
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)  # ensure unit length

        # Initialize a blank, fully transparent canvas as the final texture for this plane
        final_texture = Image.new('RGBA', original_image_size, (0, 0, 0, 0))
        painter = ImageDraw.Draw(final_texture)

        # === Core revised logic (per-point painting) ===
        # For each inlier 3D point:
        #   (1) Project to camera & sample average color in a disk around the pixel
        #   (2) Project to the plane (closest point)
        #   (3) Reproject plane point to camera; draw a disk there using the sampled color
        # This builds a mask implicitly while painting (nonzero alpha where dots were drawn).
        print("   - Painting reprojected dots from camera-sampled colors...")
        # Pre-project all to speed up (optional)
        pts_img_initial = project_points_to_2d(inlier_points_3d_original, intrinsics)

        w, h = original_image_size
        for idx, p3d in enumerate(inlier_points_3d_original):
            uv1 = pts_img_initial[idx]  # Step 1 projection
            # Skip points that are clearly outside the image bounds (with a small margin)
            if uv1[0] < -10 or uv1[0] > w + 10 or uv1[1] < -10 or uv1[1] > h + 10:
                continue

            # (1) Sample color at the camera projection
            color_rgba = _sample_disk_color_rgba(source_image, uv1, color_sample_radius)
            if color_rgba[3] == 0:
                # No valid samples—skip drawing
                continue

            # (2) Project the 3D point to the plane
            p_on_plane = _project_point_to_plane(p3d, center_c, plane_normal)

            # (3) Reproject this *plane* point to the camera
            uv2 = project_points_to_2d(p_on_plane[None, :], intrinsics)[0]

            # Draw the colored dot at the reprojected position (clamped inside image)
            cx = float(np.clip(uv2[0], 0, w - 1))
            cy = float(np.clip(uv2[1], 0, h - 1))
            painter.ellipse(
                (cx - point_draw_radius, cy - point_draw_radius, cx + point_draw_radius, cy + point_draw_radius),
                fill=color_rgba
            )

        # === Save unique texture asset for this plane ===
        texture_filename = f"plane_{plane_id}_texture.png"
        texture_filepath = assets_path / texture_filename
        final_texture.save(texture_filepath)
        print(f"   - Saved revised texture: {texture_filepath}")

        # === Build the render mesh (same OBB quad logic as original) ===
        center_local = np.array(plane_info['obb_center'])  # centered frame
        major_axis, minor_axis = axes[sorted_indices[2]], axes[sorted_indices[1]]
        major_half, minor_half = extent[sorted_indices[2]] / 2.0, extent[sorted_indices[1]] / 2.0

        v1 = center_local - major_axis * major_half + minor_axis * minor_half
        v2 = center_local + major_axis * major_half + minor_axis * minor_half
        v3 = center_local + major_axis * major_half - minor_axis * minor_half
        v4 = center_local - major_axis * major_half - minor_axis * minor_half

        corners_3d_centered = np.array([v1, v2, v3, v4])
        mesh_vertices = [v.tolist() for v in corners_3d_centered]
        mesh_faces = [[0, 1, 3], [1, 2, 3]]

        # === UVs are still computed against the full image ===
        corners_3d_original = corners_3d_centered + center_offset
        corners_2d = project_points_to_2d(corners_3d_original, intrinsics)
        uv_corners = corners_2d / np.array(original_image_size, dtype=np.float32)

        textured_planes_list.append({
            "plane_id": plane_id,
            "mesh_vertices": mesh_vertices,
            "mesh_faces": mesh_faces,
            "mesh_uvs": uv_corners.tolist(),
            "texture_path": str(Path("assets") / "planes" / texture_filename)
        })

    vignette.metadata['textured_abstractions'] = {'planes': textured_planes_list}
    vignette.save()
    print("\n--- Finished Revised Textured Plane Creation ---")