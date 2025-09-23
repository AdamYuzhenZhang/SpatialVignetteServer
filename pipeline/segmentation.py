# pipeline/segmentation.py

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2

# -------------------------
# Backend selection
# -------------------------
SAM_BACKEND = os.getenv("SAM_BACKEND", "").strip().lower()  # "coreml" | "pytorch"
if SAM_BACKEND not in ["coreml", "pytorch"]:
    raise RuntimeError(
        "Invalid or missing SAM_BACKEND environment variable. "
        "Please set it to 'coreml' or 'pytorch'.\n"
        "Example: SAM_BACKEND=coreml uvicorn main:app --reload"
    )

# Conditional imports based on the selected backend
if SAM_BACKEND == "coreml":
    # Check if we are on macOS before trying to import coremltools
    if sys.platform != "darwin":
        raise RuntimeError("CoreML backend can only be used on macOS.")
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: `coremltools` is not installed. Please run `pip install coremltools`.")
        sys.exit(1)

elif SAM_BACKEND == "pytorch":
    try:
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError:
        print("ERROR: `transformers` or `torch` not installed. Please install them to use the PyTorch backend.")
        sys.exit(1)

# -------------------------
# Shared helpers
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _uv_to_px(uv: Tuple[float, float], size_wh: Tuple[int, int]) -> Tuple[int, int]:
    u, v = float(uv[0]), float(uv[1])
    w, h = size_wh
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)
    return int(round(u * (w - 1))), int(round(v * (h - 1)))


# -------------------------
# CoreML SAM2 (3-part)
# -------------------------
_coreml_models = None
def _load_coreml_sam2():
    global _coreml_models
    if _coreml_models is not None:
        return _coreml_models

    image_encoder_path = os.getenv("COREML_IMAGE_ENCODER", "models/SAM2_1BasePlusImageEncoderFLOAT16.mlpackage")
    prompt_encoder_path = os.getenv("COREML_PROMPT_ENCODER", "models/SAM2_1BasePlusPromptEncoderFLOAT16.mlpackage")
    mask_decoder_path  = os.getenv("COREML_MASK_DECODER",  "models/SAM2_1BasePlusMaskDecoderFLOAT16.mlpackage")

    for p in [image_encoder_path, prompt_encoder_path, mask_decoder_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing CoreML SAM2 part: {p}")

    print(f"[sam2] Loading CoreML SAM2 models…")
    image_encoder = ct.models.MLModel(image_encoder_path)
    prompt_encoder = ct.models.MLModel(prompt_encoder_path)
    mask_decoder  = ct.models.MLModel(mask_decoder_path)
    _coreml_models = (image_encoder, prompt_encoder, mask_decoder)
    return _coreml_models


def _generate_coreml_sam2_logits(rgb_image_path: Path, uv_coords: List[float], output_path: Path) -> Path:
    """Run official SAM2 CoreML (3 models) with one positive point prompt."""
    image_encoder, prompt_encoder, mask_decoder = _load_coreml_sam2()

    # 1. Preprocess image → fixed size 1024×1024
    input_size = (1024, 1024)
    img = Image.open(rgb_image_path).convert("RGB")
    orig_w, orig_h = img.size
    resized = img.resize(input_size, Image.Resampling.LANCZOS)

    # 2. Image embeddings
    img_embeds = image_encoder.predict({"image": resized})

    # 3. Prompt embeddings
    px, py = _uv_to_px((uv_coords[0], uv_coords[1]), (orig_w, orig_h))
    # Scale coords to resized frame
    x = px * (input_size[0] / orig_w)
    y = py * (input_size[1] / orig_h)
    coords = np.array([[[x, y]]], dtype=np.float32)      # (1,1,2)
    labels = np.array([[1]], dtype=np.int32)             # (1,1) → positive point

    prompt_embeds = prompt_encoder.predict({"points": coords, "labels": labels})

    # 4. Mask decoder
    mask_out = mask_decoder.predict({
        "image_embedding": img_embeds["image_embedding"],
        "sparse_embedding": prompt_embeds["sparse_embeddings"],
        "dense_embedding": prompt_embeds["dense_embeddings"],
        "feats_s0": img_embeds["feats_s0"],
        "feats_s1": img_embeds["feats_s1"],
    })

    scores = mask_out["scores"]
    best_idx = np.argmax(scores)
    low_res_mask = mask_out["low_res_masks"][0, best_idx]

    # 5. Resize mask back to original image size
    mask = cv2.resize(low_res_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    _ensure_dir(Path(output_path))
    logits_path = Path(output_path) / "mask_logits.npy"
    np.save(logits_path, mask.astype(np.float32))
    print(f"[sam2] Saved CoreML SAM2 logits: {logits_path} shape={mask.shape}")
    return logits_path


# -------------------------
# PyTorch SAM
# -------------------------
_torch_model = None
_torch_processor = None
_torch_device = None
def _load_torch():
    global _torch_model, _torch_processor, _torch_device
    if _torch_model is not None:
        return _torch_model, _torch_processor, _torch_device

    repo = os.getenv("HF_SAM_REPO", "facebook/sam-vit-base")
    print(f"[sam] Loading HF model: {repo}")
    _torch_model = SamModel.from_pretrained(repo)
    _torch_processor = SamProcessor.from_pretrained(repo)
    _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    _torch_model.to(_torch_device)
    print(f"[sam] Torch device: {_torch_device}")
    return _torch_model, _torch_processor, _torch_device


def _generate_pytorch_sam_logits(rgb_image_path: Path, uv_coords: List[float], output_path: Path) -> Path:
    model, processor, device = _load_torch()

    image = Image.open(rgb_image_path).convert("RGB")
    w, h = image.size
    px, py = _uv_to_px((uv_coords[0], uv_coords[1]), (w, h))
    input_points = [[[int(px), int(py)]]]

    inputs = processor(image, input_points=input_points, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    masks = outputs.pred_masks.detach().to("cpu")
    masks = torch.nn.functional.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    logits = masks[0, 0].numpy().astype(np.float32)

    _ensure_dir(Path(output_path))
    logits_path = Path(output_path) / "mask_logits.npy"
    np.save(logits_path, logits)
    print(f"[sam] Saved Torch logits: {logits_path} shape={logits.shape}")
    return logits_path


# -------------------------
# Public API
# -------------------------
def generate_sam_logits(rgb_image_path: Path, uv_coords: List[float], output_path: Path) -> Path:
    if SAM_BACKEND == "coreml":
        return _generate_coreml_sam2_logits(rgb_image_path, uv_coords, output_path)
    elif SAM_BACKEND == "pytorch":
        return _generate_pytorch_sam_logits(rgb_image_path, uv_coords, output_path)
    else:
        raise RuntimeError("Set SAM_BACKEND to 'coreml' or 'pytorch'")


def apply_threshold_to_logits(logits_path: Path, threshold: float, output_path: Path) -> Path:
    _ensure_dir(Path(output_path))
    logits = np.load(logits_path)
    mask = (logits > float(threshold)).astype(np.uint8) * 255
    mask_path = Path(output_path) / "mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)
    print(f"[sam] Wrote mask: {mask_path}")
    return mask_path