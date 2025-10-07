# metric_depth.py

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


# -------------------------
# Model architecture configs
# -------------------------
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# -------------------------
# Presets for quick switching
# -------------------------
# scene -> (dataset tag, default max_depth, checkpoint filename template)
PRESETS = {
    'indoor':  ('hypersim', 20.0, 'depth_anything_v2_metric_hypersim_{encoder}.pth'),
    'outdoor': ('vkitti',   80.0, 'depth_anything_v2_metric_vkitti_{encoder}.pth'),
}
SceneStr = Literal['indoor', 'outdoor']
DeviceStr = Literal['cuda', 'mps', 'cpu']

def pick_device() -> DeviceStr:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@dataclass
class MetricDepthConfig:
    # Core knobs
    encoder: Literal['vits', 'vitb', 'vitl'] = 'vitl'
    dataset: Literal['hypersim', 'vkitti'] = 'hypersim'  # bookkeeping only
    max_depth: float = 20.0
    input_size: int = 518

    # Files / runtime
    checkpoint_path: Optional[str] = None
    checkpoints_dir: str = 'depth_anything_v2/checkpoints'
    device: Optional[DeviceStr] = None

    def resolved_checkpoint(self) -> str:
        """Build a default checkpoint path if one isn't provided."""
        if self.checkpoint_path:
            return self.checkpoint_path
        # infer from dataset + encoder
        if self.dataset == 'hypersim':
            fname = f'depth_anything_v2_metric_hypersim_{self.encoder}.pth'
        elif self.dataset == 'vkitti':
            fname = f'depth_anything_v2_metric_vkitti_{self.encoder}.pth'
        else:
            raise ValueError(f'Unknown dataset {self.dataset}')
        return os.path.join(self.checkpoints_dir, fname)


class DepthAnythingMetric:
    """
    Metric Depth Anything V2 (returns meters).
    Provides both config-driven and preset-driven constructors.
    """
    def __init__(self, cfg: MetricDepthConfig):
        self.cfg = cfg
        self.device = cfg.device or pick_device()

        if cfg.encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder {cfg.encoder}. Choose from {list(MODEL_CONFIGS.keys())}.")

        model_kwargs = {**MODEL_CONFIGS[cfg.encoder], 'max_depth': cfg.max_depth}
        self.net = DepthAnythingV2(**model_kwargs)

        ckpt = cfg.resolved_checkpoint()
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        state = torch.load(ckpt, map_location='cpu')
        self.net.load_state_dict(state)
        self.net = self.net.to(self.device).eval()

    # ---------
    # Factories
    # ---------
    @classmethod
    def from_preset(
        cls,
        scene: SceneStr = 'indoor',
        *,
        encoder: Literal['vits', 'vitb', 'vitl'] = 'vitl',
        input_size: int = 518,
        checkpoints_dir: str = 'depth_anything_v2/checkpoints',
        max_depth: Optional[float] = None,
        device: Optional[DeviceStr] = None,
        checkpoint_path: Optional[str] = None,
    ) -> 'DepthAnythingMetric':
        """
        Build with a single switch: scene='indoor'|'outdoor'.
        You can still override encoder, max_depth, etc.
        """
        if scene not in PRESETS:
            raise ValueError(f"scene must be one of {list(PRESETS.keys())}")
        dataset_tag, default_md, template = PRESETS[scene]
        md = float(max_depth) if max_depth is not None else default_md

        if checkpoint_path is None:
            fname = template.format(encoder=encoder)
            checkpoint_path = os.path.join(checkpoints_dir, fname)

        cfg = MetricDepthConfig(
            encoder=encoder,
            dataset=dataset_tag,
            max_depth=md,
            input_size=input_size,
            checkpoint_path=checkpoint_path,
            checkpoints_dir=checkpoints_dir,
            device=device,
        )
        return cls(cfg)

    # -------------
    # Inference APIs
    # -------------
    def infer_bgr(self, image_bgr: np.ndarray, input_size: Optional[int] = None) -> np.ndarray:
        """
        Args:
            image_bgr: uint8 (H,W,3) BGR
            input_size: optional override of resize short side
        Returns:
            depth_m: float32 (H,W) in meters
        """
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be uint8 HxWx3 (BGR)")

        size = int(input_size) if input_size is not None else int(self.cfg.input_size)
        with torch.no_grad():
            depth_m = self.net.infer_image(image_bgr, size)  # numpy HxW float32 in meters
        return depth_m

    def infer_rgb(self, image_rgb: np.ndarray, input_size: Optional[int] = None) -> np.ndarray:
        """Accept RGB uint8 and convert to BGR internally."""
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be uint8 HxWx3 (RGB)")
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.infer_bgr(image_bgr, input_size=input_size)


# --------------------------------------
# Convenience function with model caching
# --------------------------------------
# Avoid reloading weights on every call; cache by (scene, encoder, device, max_depth, input_size, ckpt_path)
_MODEL_CACHE: Dict[Tuple, DepthAnythingMetric] = {}

def estimate_depth_metric(
    image: np.ndarray,
    *,
    scene: SceneStr = 'indoor',                # 'indoor' (Hypersim, 20m) or 'outdoor' (VKITTI, 80m)
    encoder: Literal['vits', 'vitb', 'vitl'] = 'vitl',
    input_size: int = 518,
    checkpoints_dir: str = 'depth_anything_v2/checkpoints',
    max_depth: Optional[float] = None,
    device: Optional[DeviceStr] = None,
    checkpoint_path: Optional[str] = None,
    image_is_bgr: bool = False,                # True if image came from cv2.imread directly
) -> np.ndarray:
    """
    One-call API: give me an image, I return depth in meters.
    Uses an internal model cache keyed by config to avoid reloads.
    """
    key = (
        scene, encoder, device or pick_device(),
        float(max_depth) if max_depth is not None else None,
        int(input_size),
        checkpoint_path or os.path.join(
            checkpoints_dir,
            PRESETS[scene][2].format(encoder=encoder)
        )
    )

    model = _MODEL_CACHE.get(key)
    if model is None:
        model = DepthAnythingMetric.from_preset(
            scene=scene,
            encoder=encoder,
            input_size=input_size,
            checkpoints_dir=checkpoints_dir,
            max_depth=max_depth,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        _MODEL_CACHE[key] = model

    if image_is_bgr:
        return model.infer_bgr(image, input_size=None)
    else:
        return model.infer_rgb(image, input_size=None)