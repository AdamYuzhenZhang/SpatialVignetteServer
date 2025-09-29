import json
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image

from pipeline.vignette_data import ProcessedVignette

def create_masked_point_cloud(vignette_path: Path):
    """
    Creates a ProcessedVignette object from raw data, including points, colors,
    and confidence values, and saves it to a single .npz file.

    This function is the first step in the pipeline, converting the raw capture
    data into our standardized ProcessedVignette format.
    """
    # 1. Define and validate all necessary file paths
    rgb_path = vignette_path / "rgb.png"
    depth_path = vignette_path / "depth.bin"
    confidence_path = vignette_path / "confidence.bin"
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
            capture_metadata = json.load(f)
        
        depth_height, depth_width = capture_metadata['resolution'][1], capture_metadata['resolution'][0]
        target_size = (depth_width, depth_height)

        color_img_raw = Image.open(rgb_path).convert("RGB")
        mask_img_raw = Image.open(mask_path).convert("L")
        
        depth_data = np.fromfile(depth_path, dtype=np.float32)
        depth_image_np = depth_data.reshape((depth_height, depth_width))

        intrinsics_matrix = np.array(capture_metadata['camera_intrinsics']['columns']).T

        confidence_map_np = None
        if confidence_path.exists():
            confidence_data = np.fromfile(confidence_path, dtype=np.uint8)
            confidence_map_np = confidence_data.reshape((depth_height, depth_width))
        else:
            print("Warning: Confidence file not found. Confidence values will be missing.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # 3. Scale camera intrinsics based on the final depth resolution
    original_width, original_height = color_img_raw.size
    x_scale, y_scale = depth_width / original_width, depth_height / original_height
    fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    scaled_fx, scaled_fy, scaled_cx, scaled_cy = fx * x_scale, fy * y_scale, cx * x_scale, cy * y_scale
    
    # 4. Resize images to match depth map dimensions
    color_img_resized = color_img_raw.resize(target_size, Image.Resampling.BILINEAR)
    mask_img_resized = mask_img_raw.resize(target_size, Image.Resampling.NEAREST)

    # 5. Apply the resized mask to the depth data to isolate the object of interest
    mask_np = np.array(mask_img_resized) > 127
    depth_image_np[~mask_np] = 0.0

    # 6. Create Open3D structures for point cloud generation
    o3d_color = o3d.geometry.Image(np.array(color_img_resized))
    o3d_depth = o3d.geometry.Image(depth_image_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        depth_width, depth_height, scaled_fx, scaled_fy, scaled_cx, scaled_cy
    )

    # 7. Generate the point cloud and normalize its position by centering it
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)
    center = pcd.get_center()
    pcd.translate(-center)
    print(f"Generated and centered point cloud with {len(pcd.points)} points.")
    
    # --- NEW: Store the centering vector in the metadata ---
    # This is crucial for correctly un-doing the translation during texturing.
    capture_metadata['center_offset'] = center.tolist()
    
    # 8. Extract final data arrays for our ProcessedVignette
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    attributes = {}
    
    if confidence_map_np is not None:
        valid_confidences = confidence_map_np[mask_np]
        if len(valid_confidences) == len(points):
            attributes['confidence'] = valid_confidences
        else:
            print(
                f"Warning: Point count ({len(points)}) and confidence count "
                f"({len(valid_confidences)}) mismatch. Skipping confidence data."
            )

    processed_vignette = ProcessedVignette(
        points=points,
        colors=colors,
        metadata={"capture_metadata": capture_metadata},
        **attributes
    )
    
    # 9. Save the entire, self-contained object to a single .npz file
    results_path = vignette_path / "results"
    results_path.mkdir(exist_ok=True)
    output_npz_path = results_path / "raw_vignette.npz"
    processed_vignette.save(output_npz_path)

    return output_npz_path

