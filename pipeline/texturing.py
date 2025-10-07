"""
texturing.py - Plane-driven textured scraps (RANSAC plane + OBB footprint)

This module uses planes detected by pyransac3D (stored as 'equation' ax+by+cz+d=0)
to define the true plane geometry. The plane is bounded by the OBB *in-plane*
rectangle (taken from the OBB of the inliers) to create a finite "scrap."
Colors are gathered from the RGB image at each point's original camera projection,
then placed at the reprojected location of the closest point on the plane.

Outputs per-plane:
- A PNG texture with painted dots (transparent background elsewhere)
- A quad mesh aligned with the OBB in-plane axes (on the plane)
- UVs computed against the full image size (no cropping)

Main function:
    create_textured_planes(vignette, rgb_image_path, output_dir, point_radius=5)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import open3d as o3d

# Import our custom data structure
from .vignette_data import ProcessedVignette


# --------------------------- Camera / Math helpers ---------------------------

def project_points_to_2d(points_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Standard pinhole projection (no extrinsics):
        [u, v, 1]^T ~ K [X, Y, Z]^T,  u = x/z, v = y/z

    Args:
        points_3d: (N, 3) in original/camera frame
        intrinsics: (3,3) camera intrinsics matrix

    Returns:
        (N, 2) pixel coordinates
    """
    P = intrinsics @ points_3d.T
    d = P[2, :] + 1e-9
    u = P[0, :] / d
    v = P[1, :] / d
    return np.vstack((u, v)).T


def plane_equation_to_point_normal(eq: np.ndarray, fallback_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn ax+by+cz+d=0 into a (point_on_plane, unit_normal).
    If fallback_point is provided, we choose the closest point on the plane to it
    for numerical stability.

    Args:
        eq: (4,) plane equation [a, b, c, d]
        fallback_point: (3,) point in space to anchor plane point (optional)

    Returns:
        (p0, n) where p0 is a point on the plane and n is unit normal
    """
    a, b, c, d = eq
    n = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(n) + 1e-12
    n = n / norm

    if fallback_point is None:
        # choose point along the normal where plane crosses
        # pick a coordinate to avoid division by near-zero
        # e.g., set z = 0 => solve ax+by+d=0 if c not small, prefer axis with largest |component|
        idx = int(np.argmax(np.abs(n)))
        p = np.zeros(3, dtype=np.float64)
        # Solve for that axis variable from n·p + d/norm = 0 => n·p = -d/norm
        # Set two coordinates to 0; solve for the chosen axis:
        p[idx] = - (d / norm) / (n[idx] + 1e-12)
        return p, n
    else:
        # closest point on plane to fallback_point:
        # p_closest = f - ((n·f + d/norm) * n)
        f = np.asarray(fallback_point, dtype=np.float64)
        signed = (np.dot(n, f) + d / norm)
        return f - signed * n, n


def project_point_to_plane(p: np.ndarray, p0: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Closest orthogonal projection of p onto plane (p0, n)."""
    return p - np.dot(p - p0, n) * n


def clamp_point_to_obb_on_plane(p_on_plane: np.ndarray,
                                obb_center_world: np.ndarray,
                                u_axis: np.ndarray, v_axis: np.ndarray,
                                u_half: float, v_half: float) -> np.ndarray:
    """
    Keep an on-plane point inside the plane's OBB footprint by clamping along
    in-plane axes u and v. u_axis and v_axis must be orthonormal in the plane.
    """
    rel = p_on_plane - obb_center_world
    u = float(np.dot(rel, u_axis))
    v = float(np.dot(rel, v_axis))
    u = np.clip(u, -u_half, u_half)
    v = np.clip(v, -v_half, v_half)
    return obb_center_world + u * u_axis + v * v_axis


def sample_disk_color_rgba(img: Image.Image, uv: np.ndarray, radius: int) -> tuple:
    """
    Average RGBA over a disk centered at uv with given radius (pixels).
    Returns (R,G,B,A); A=0 => no valid sample (e.g., off-image).
    """
    w, h = img.size
    cx, cy = float(uv[0]), float(uv[1])
    x_min = max(int(np.floor(cx - radius)), 0)
    x_max = min(int(np.ceil (cx + radius)), w - 1)
    y_min = max(int(np.floor(cy - radius)), 0)
    y_max = min(int(np.ceil (cy + radius)), h - 1)
    if x_min > x_max or y_min > y_max:
        return (0, 0, 0, 0)

    crop = img.crop((x_min, y_min, x_max + 1, y_max + 1)).convert("RGBA")
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    ccx, ccy = cx - x_min, cy - y_min
    draw.ellipse((ccx - radius, ccy - radius, ccx + radius, ccy + radius), fill=255)

    crop_np = np.asarray(crop, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.uint8)
    sel = mask_np > 0
    if not np.any(sel):
        return (0, 0, 0, 0)

    vals = crop_np[sel].reshape(-1, 4)
    mean = vals.mean(axis=0)
    r, g, b, a = [int(np.clip(v, 0, 255)) for v in mean]
    return (r, g, b, 255)  # opaque paint; change if you want soft alpha


# --------------------------- Core texturing pipeline ---------------------------

def create_textured_planes(
    vignette: "ProcessedVignette",
    rgb_image_path: str,
    output_dir: str,
    point_radius: int = 5,
    color_sample_radius: Optional[int] = None,
    clamp_to_obb: bool = True,
    z_min: float = 1e-6
) -> None:
    """
    Generates masked textures & meshes for dominant planes using the *RANSAC plane equation*
    as the true plane geometry, and the OBB only to define a finite in-plane rectangle.

    Revised dot logic:
        For each inlier 3D point p:
          (1) Project p to the camera image; sample average color in a disk.
          (2) Project p to its closest point on the fitted plane (orthogonal).
              Optionally clamp to OBB rectangle (keeps dots on scrap).
          (3) Reproject that on-plane point to the camera; if Z<=0, skip.
              Paint a disk of radius 'point_radius' with the sampled color.

    Mesh quad:
        - Built on the plane using OBB in-plane axes (u_axis, v_axis) and extents.
        - The normal used for placement is from the plane equation (not OBB).
        - UVs computed against the full original image (no cropping).

    Args:
        vignette: ProcessedVignette with 'planes' abstractions. Each plane must include:
                    - 'equation': [a,b,c,d] from pyransac3D
                    - 'point_indices': list of inlier point indices
                    - OBB fields: 'obb_center', 'obb_rotation', 'obb_extent' (for bounds)
        rgb_image_path: File path to the original RGBA/RGB image (used for sampling and canvas size).
        output_dir: Base directory where assets will be saved.
        point_radius: Radius (px) of painted disks in the final texture.
        color_sample_radius: Optional sampling radius (px). Defaults to point_radius if None.
        clamp_to_obb: Clamp plane points to the OBB rectangle footprint.
        z_min: Minimum Z for visibility in camera frame (skip if <= z_min).
    """
    if color_sample_radius is None:
        color_sample_radius = point_radius

    print("--- Starting Plane-Driven Textured Plane Creation ---")
    vignette.clear_abstractions('textured_planes', auto_save=False)

    # === Load planes and camera ===
    planes = vignette.get_abstractions('planes')
    if not planes:
        print("   - No planes found. Aborting.")
        return

    capture_meta = vignette.metadata.get('capture_metadata', {})
    intrinsics = np.array(capture_meta.get('camera_intrinsics', {}).get('columns', np.identity(3))).T
    center_offset = np.array(capture_meta.get('center_offset', [0, 0, 0]))

    try:
        source_image = Image.open(rgb_image_path).convert("RGBA")
        im_w, im_h = source_image.size
    except Exception as e:
        print(f"   - ERROR: Could not read source image at {rgb_image_path}. Error: {e}")
        return

    if np.array_equal(intrinsics, np.identity(3)) or np.allclose(center_offset, [0, 0, 0]):
        print("   - ERROR: Missing camera intrinsics or center_offset; projections may be invalid.")
        return

    # === Prepare asset folder ===
    assets_path = Path(output_dir) / "assets" / "planes"
    assets_path.mkdir(parents=True, exist_ok=True)

    textured_planes_list: List[Dict[str, Any]] = []

    # === Loop over planes ===
    for plane in planes:
        plane_id = plane.get('plane_id')
        print(f"\n--- Processing Plane {plane_id} ---")

        # Required fields:
        if not all(k in plane for k in ['equation', 'point_indices', 'obb_center', 'obb_rotation', 'obb_extent']):
            print("   - Skipping: plane missing 'equation', 'point_indices', or OBB fields.")
            continue

        eq = np.array(plane['equation'], dtype=np.float64)  # [a,b,c,d]
        inlier_idx = plane['point_indices']

        # Inlier points (centered coords -> original/camera coords)
        pts_centered = vignette.points[inlier_idx]
        pts = pts_centered + center_offset

        # Build plane geometry from equation
        # Use OBB center as fallback point to get a stable point on the plane
        obb_center_world = np.array(plane['obb_center']) + center_offset
        p0, n = plane_equation_to_point_normal(eq, fallback_point=obb_center_world)

        # OBB frame for in-plane bounds (u = major, v = minor by extent)
        R = np.array(plane['obb_rotation'])
        extent = np.array(plane['obb_extent'])
        axes = [R[:, i] for i in range(3)]
        order = np.argsort(extent)          # 0=thin (normal), 1=minor=v, 2=major=u
        thin_axis = axes[order[0]]
        v_axis = axes[order[1]]
        u_axis = axes[order[2]]

        # Orthonormalize in-plane axes (in case numerical drift)
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)

        # Align plane normal to OBB thin direction for consistency
        thin_axis = thin_axis / (np.linalg.norm(thin_axis) + 1e-12)
        if np.dot(n, thin_axis) < 0:
            n = -n

        u_half = float(extent[order[2]] / 2.0)
        v_half = float(extent[order[1]] / 2.0)

        # Diagnostics (optional): mean distance of inliers to plane
        d_signed = np.dot(pts - p0, n)
        print(f"   - Plane residual abs mean/max: {np.mean(np.abs(d_signed)):.4g} / {np.max(np.abs(d_signed)):.4g}")

        # === Build the painted texture for this plane ===
        canvas = Image.new('RGBA', (im_w, im_h), (0, 0, 0, 0))
        painter = ImageDraw.Draw(canvas)

        # Step 1: project points to camera and sample colors on source image
        uv0 = project_points_to_2d(pts, intrinsics)

        for i, p in enumerate(pts):
            uv = uv0[i]
            # quick bounds check with small margin; full clamp later
            if uv[0] < -10 or uv[0] > im_w + 10 or uv[1] < -10 or uv[1] > im_h + 10:
                continue

            color = sample_disk_color_rgba(source_image, uv, color_sample_radius)
            if color[3] == 0:
                continue

            # Step 2: project to closest point on the plane
            q = project_point_to_plane(p, p0, n)

            # Optional: clamp to plane's OBB rectangle footprint
            if clamp_to_obb:
                q = clamp_point_to_obb_on_plane(q, obb_center_world, u_axis, v_axis, u_half, v_half)

            # Skip if behind the camera
            if q[2] <= z_min:
                continue

            # Step 3: reproject plane point and paint
            uv2 = project_points_to_2d(q[None, :], intrinsics)[0]
            cx = float(np.clip(uv2[0], 0, im_w - 1))
            cy = float(np.clip(uv2[1], 0, im_h - 1))
            painter.ellipse(
                (cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius),
                fill=color
            )

        # Save texture
        texture_filename = f"plane_{plane_id}_texture.png"
        texture_filepath = assets_path / texture_filename
        canvas.save(texture_filepath)
        print(f"   - Saved texture: {texture_filepath}")

        # === Construct the mesh quad on the plane using OBB in-plane axes ===
        # Quad corners in centered (vignette) coords: use OBB axes/extents, then rely
        # on your renderer's centered frame. (We keep consistency with your prior mesh API.)
        center_local = np.array(plane['obb_center'])  # centered frame
        # IMPORTANT: u_axis, v_axis are defined in *original* frame; for the centered frame,
        # we assume axes are the same directions (vignette points were just translated by center_offset).
        # If your pipeline rotates frames, inject the proper transform here.
        v1 = center_local - u_axis * u_half + v_axis * v_half
        v2 = center_local + u_axis * u_half + v_axis * v_half
        v3 = center_local + u_axis * u_half - v_axis * v_half
        v4 = center_local - u_axis * u_half - v_axis * v_half

        corners_3d_centered = np.array([v1, v2, v3, v4])
        mesh_vertices = [v.tolist() for v in corners_3d_centered]
        mesh_faces = [[0, 1, 3], [1, 2, 3]]

        # UVs against full image
        corners_3d_original = corners_3d_centered + center_offset
        uv_corners = project_points_to_2d(corners_3d_original, intrinsics) / np.array([im_w, im_h], dtype=np.float32)

        # Accumulate metadata
        textured_planes_list.append({
            "plane_id": plane_id,
            "mesh_vertices": mesh_vertices,
            "mesh_faces": mesh_faces,
            "mesh_uvs": uv_corners.tolist(),
            "texture_path": str(Path("assets") / "planes" / texture_filename),
            # Optional debug fields:
            "plane_point_world": p0.tolist(),
            "plane_normal_world": n.tolist(),
            "obb_center_world": obb_center_world.tolist(),
            "u_axis_world": u_axis.tolist(),
            "v_axis_world": v_axis.tolist(),
            "u_half": u_half,
            "v_half": v_half,
        })

    vignette.metadata['textured_abstractions'] = {'planes': textured_planes_list}
    vignette.save()
    print("\n--- Finished Plane-Driven Textured Plane Creation ---")