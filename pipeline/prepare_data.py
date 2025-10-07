# Prepares data for preprocessing
# For image only captures, wrap photo and generated depth into correct formats.

from __future__ import annotations
import json, math, warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ExifTags

# Check if heif is supported
_HEIF_ENABLED = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_ENABLED = True
except Exception:
    warnings.warn(
        "pillow-heif not installed; HEIF/HEIC images may fail to open. "
        "Install with: pip install pillow-heif",
        stacklevel=1,
    )

def load_image_rgb_any(img_path: str | Path) -> np.ndarray:
    """
    Load an image (HEIC/HEIF/JPEG/PNG) and return an RGB NumPy array (uint8).
    - Requires pillow-heif for HEIC/HEIF support.
    """
    img = Image.open(img_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

# ---------------------------
# EXIF helpers (HFOV recovery)
# ---------------------------
_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_TAG_FOCAL_LENGTH_35MM = _EXIF_TAGS.get("FocalLengthIn35mmFilm")   # 0xA405
_TAG_FOCAL_LENGTH      = _EXIF_TAGS.get("FocalLength")             # 0x920A
_TAG_FP_XRES           = _EXIF_TAGS.get("FocalPlaneXResolution")   # 0xA20E
_TAG_FP_UNIT           = _EXIF_TAGS.get("FocalPlaneResolutionUnit")# 0xA210
# Unit values per EXIF spec: 2=inches, 3=centimeters ; some files use 4=millimeters
_UNIT_TO_MM = {2: 25.4, 3: 10.0, 4: 1.0}

def _rational_to_float(val):
    """EXIF often stores rationals; convert to float robustly."""
    try:
        if hasattr(val, "numerator") and hasattr(val, "denominator"):
            den = float(val.denominator) if val.denominator else 1.0
            return float(val.numerator) / den
        if isinstance(val, (tuple, list)) and len(val) == 2:
            num, den = val
            den = float(den) if den else 1.0
            return float(num) / den
        return float(val)
    except Exception:
        return None

def _get_exif(img: Image.Image):
    """Return PIL Exif object or empty dict-like; handles HEIF when possible."""
    try:
        exif = img.getexif()
        # Some HEIF images store EXIF bytes in img.info["exif"]; getexif() should parse that.
        return exif if exif else {}
    except Exception:
        return {}

def try_hfov_from_exif(image_path: str | Path, width_px: int) -> Tuple[Optional[float], str]:
    """
    Attempt to derive horizontal FOV (degrees) from EXIF.
    Returns (hfov_deg or None, 'method description').
    """
    method = "no_exif"
    try:
        img = Image.open(image_path)
        exif = _get_exif(img)
        if not exif:
            return None, "no_exif"

        # 1) Best: FocalLengthIn35mmFilm (35mm equivalent) → HFOV from 36mm sensor width
        if _TAG_FOCAL_LENGTH_35MM in exif:
            f35 = _rational_to_float(exif[_TAG_FOCAL_LENGTH_35MM])
            if f35 and f35 > 0:
                hfov_rad = 2.0 * math.atan(36.0 / (2.0 * f35))  # 36mm full-frame width
                return math.degrees(hfov_rad), "focal_length_35mm"

        # 2) Next: true focal length + focal plane pixel density => sensor width
        f_mm = _rational_to_float(exif[_TAG_FOCAL_LENGTH]) if _TAG_FOCAL_LENGTH in exif else None
        xres = _rational_to_float(exif[_TAG_FP_XRES]) if _TAG_FP_XRES in exif else None
        unit = int(exif[_TAG_FP_UNIT]) if _TAG_FP_UNIT in exif and exif[_TAG_FP_UNIT] is not None else None
        unit_mm = _UNIT_TO_MM.get(unit)

        if f_mm and f_mm > 0 and xres and xres > 0 and unit_mm:
            sensor_width_mm = (width_px / xres) * unit_mm
            hfov_rad = 2.0 * math.atan(sensor_width_mm / (2.0 * f_mm))
            return math.degrees(hfov_rad), "focal_length+focal_plane_res"

        return None, "exif_incomplete"
    except Exception:
        return None, "exif_error"

# ---------------------------
# Intrinsics from HFOV
# ---------------------------

def make_intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """
    Build a 3x3 pinhole intrinsics K from horizontal FOV (deg).
    Assumes square pixels and principal point at image center.
    """
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def estimate_intrinsics_from_image(
    image_path: str | Path,
    default_hfov_deg: float = 60.0
) -> Tuple[np.ndarray, float, str, Tuple[int, int]]:
    """
    Try to estimate intrinsics from EXIF. If unavailable, fall back to default HFOV.
    Returns: (K, used_hfov_deg, method, (width, height))
    method in {'focal_length_35mm','focal_length+focal_plane_res','exif_incomplete','no_exif','exif_error','fallback(...)'}
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    hfov_deg, method = try_hfov_from_exif(image_path, width)
    if hfov_deg is None:
        hfov_deg = float(default_hfov_deg)
        method = f"fallback({default_hfov_deg}deg)"

    K = make_intrinsics_from_hfov(width, height, hfov_deg)
    return K, hfov_deg, method, (width, height)

# ---------------------------
# Saver: rgb.png, depth.bin, metadata.json, (confidence.bin)
# ---------------------------

def save_vignette_files(
    out_dir: str | Path,
    *,
    rgb_image_path: str | Path,
    depth_m: np.ndarray,                       # HxW float32 meters
    intrinsics_K: Optional[np.ndarray] = None, # if None -> EXIF or fallback
    subject_uv: Tuple[float, float] = (0.5, 0.5),
    confidence: Optional[np.ndarray] = None,   # HxW float32 [optional]
    default_hfov_deg: float = 60.0,            # used only if EXIF insufficient & intrinsics_K=None
):
    """
    Writes (always emits a PNG even if input is HEIC/HEIF):
      - rgb.png
      - depth.bin (float32 little-endian, row-major H*W)
      - metadata.json (resolution, camera_intrinsics.columns, subject_uv)
      - confidence.bin (optional, same dtype/layout as depth)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load image with PIL (HEIF supported if pillow-heif is installed)
    img = Image.open(rgb_image_path).convert("RGB")
    width, height = img.size
    img_png_path = out / "rgb.png"
    img.save(img_png_path)
    print(f"- RGB image saved to {img_png_path}")

    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be HxW; got shape {depth_m.shape}")
    depth_h, depth_w = depth_m.shape
    print(f"- Depth image size {depth_w} x {depth_h}")

    # Depth size check (resize if mismatched)
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be HxW; got shape {depth_m.shape}")
    if depth_m.shape != (height, width):
        # Bicubic resize (still meters; interpolation smooths)
        depth_m = np.asarray(
            Image.fromarray(depth_m.astype(np.float32), mode="F").resize((width, height), Image.BICUBIC),
            dtype=np.float32,
        )

    # Save depth.bin (float32 little-endian)
    depth_le = np.asarray(depth_m, dtype="<f4")
    depth_path = out / "depth.bin"
    depth_path.write_bytes(depth_le.tobytes(order="C"))
    print(f"- Depth map saved to {depth_path}")

    # Save optional confidence.bin
    if confidence is not None:
        if confidence.shape != (height, width):
            confidence = np.asarray(
                Image.fromarray(confidence.astype(np.float32), mode="F").resize((width, height), Image.BICUBIC),
                dtype=np.float32,
            )
        conf_le = np.asarray(confidence, dtype="<f4")
        (out / "confidence.bin").write_bytes(conf_le.tobytes(order="C"))
        print(f"- Depth map saved to {out / "confidence.bin"}")

    # Intrinsics: provided or estimated (EXIF→HFOV→K)
    if intrinsics_K is not None:
        K = np.asarray(intrinsics_K, dtype=np.float64)
        method = "provided_K"
        used_hfov = None
    else:
        K, used_hfov, method, _ = estimate_intrinsics_from_image(rgb_image_path, default_hfov_deg)

    # Your schema stores columns of K (so JSON has K^T)
    columns = K.T.tolist()

    metadata = {
        "resolution": [int(depth_w), int(depth_h)],
        "camera_intrinsics": {"columns": columns},
        "subject_uv": [float(subject_uv[0]), float(subject_uv[1])],
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        print(f"- Metadata saved to {out / "metadata.json"}")


# ---------------------------
# Convenience wrapper
# ---------------------------

def vignette_data_from_image_and_depth(
    img_path: str | Path,
    depth_m: np.ndarray,
    out_dir: str | Path,
    *,
    intrinsics_K: Optional[np.ndarray] = None,
    subject_uv=(0.5, 0.5),
    confidence: Optional[np.ndarray] = None,
    default_hfov_deg: float = 60.0,
):
    """
    Convenience wrapper: produce rgb.png, depth.bin, metadata.json (and confidence.bin).
    """
    save_vignette_files(
        out_dir,
        rgb_image_path=img_path,
        depth_m=depth_m,
        intrinsics_K=intrinsics_K,
        subject_uv=subject_uv,
        confidence=confidence,
        default_hfov_deg=default_hfov_deg,
    )
    print("- Done")