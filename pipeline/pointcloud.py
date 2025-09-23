import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

def create_masked_point_cloud(vignette_path: Path):
    """
    Creates a 3D point cloud from vignette data, filtered by a segmentation mask,
    and centers it at the origin.

    This function handles resolution mismatches by scaling the camera intrinsics and
    resizing images to match the depth map. It then reconstructs the 3D scene,
    transforms it using the camera extrinsics to get the correct orientation,
    and finally centers the object at (0,0,0) for easy use as a modular asset.

    Args:
        vignette_path: The path to the directory containing the vignette data.

    Returns:
        The file path to the saved point cloud (.ply file), or None if an error occurred.
    """
    # 1. Define and validate all necessary file paths
    rgb_path = vignette_path / "rgb.png"
    depth_path = vignette_path / "depth.bin"
    metadata_path = vignette_path / "metadata.json"
    mask_path = vignette_path / "results" / "mask.png"

    required_files = [rgb_path, depth_path, metadata_path, mask_path]
    for p in required_files:
        if not p.exists():
            print(f"Error: Missing required file: {p}")
            return None

    print(f"Processing point cloud for: {vignette_path.name}")

    try:
        # 2. Load all data
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        depth_height, depth_width = metadata['resolution'][1], metadata['resolution'][0]
        target_size = (depth_width, depth_height)

        color_img_raw = Image.open(rgb_path).convert("RGB")
        mask_img_raw = Image.open(mask_path).convert("L")

        depth_data = np.fromfile(depth_path, dtype=np.float32)
        depth_image_np = depth_data.reshape((depth_height, depth_width))

        intrinsics_matrix = np.array(metadata['camera_intrinsics']['columns']).T
        # Extrinsics are usually saved as a 4x4 matrix
        # extrinsics_matrix = np.array(metadata['camera_extrinsics'])

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # 3. FIX: Scale camera intrinsics to match the target (depth) resolution
    # The original intrinsics are calibrated for the original (high-res) RGB image.
    original_width, original_height = color_img_raw.size
    x_scale = depth_width / original_width
    y_scale = depth_height / original_height

    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

    scaled_fx = fx * x_scale
    scaled_fy = fy * y_scale
    scaled_cx = cx * x_scale
    scaled_cy = cy * y_scale
    
    print(f"Scaled intrinsics (fx, fy) from ({fx:.2f}, {fy:.2f}) to ({scaled_fx:.2f}, {scaled_fy:.2f})")

    # 4. Resize images to match depth map dimensions
    color_img_resized = color_img_raw.resize(target_size, Image.Resampling.BILINEAR)
    mask_img_resized = mask_img_raw.resize(target_size, Image.Resampling.NEAREST)

    # 5. Apply the resized mask to the depth data
    mask_np = np.array(mask_img_resized) > 127
    depth_image_np[~mask_np] = 0.0

    # 6. Create Open3D structures with SCALED intrinsics
    o3d_color = o3d.geometry.Image(np.array(color_img_resized))
    o3d_depth = o3d.geometry.Image(depth_image_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=depth_width, height=depth_height,
        fx=scaled_fx, fy=scaled_fy,
        cx=scaled_cx, cy=scaled_cy
    )

    # 7. Generate the point cloud in Camera Space
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)
    
    # We are not using extrinsics to place it in the world, as we want a modular asset.
    # If you needed to, you would apply the transform like this:
    # pcd.transform(extrinsics_matrix)

    # 8. BEST PRACTICE: Center the point cloud around its own geometric center
    center = pcd.get_center()
    pcd.translate(-center)
    print(f"Generated and centered point cloud with {len(pcd.points)} points.")
    
    # 9. Save the final, centered point cloud
    results_path = vignette_path / "results"
    results_path.mkdir(exist_ok=True)
    output_ply_path = results_path / "masked_point_cloud.ply"
    o3d.io.write_point_cloud(str(output_ply_path), pcd, write_ascii=True)
    print(f"Saved masked point cloud to: {output_ply_path}")

    return output_ply_path

