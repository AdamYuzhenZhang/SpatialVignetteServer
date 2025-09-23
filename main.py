# main.py

import uuid
import json
import shutil
import logging
import base64
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

# Import the core logic
import schemas
from pipeline import segmentation

# --- Configuration ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- FastAPI Application ---
app = FastAPI(
    title="Spatial Vignette Server",
    description="Processes 3D captures, runs segmentation, and prepares data for abstraction."
)

def get_vignette_path(vignette_id: str) -> Path:
    """Dependency that provides a valid, existing vignette path."""
    vignette_path = DATA_DIR / vignette_id
    if not vignette_path.exists():
        logging.error(f"Vignette not found: {vignette_id}")
        raise HTTPException(status_code=404, detail=f"Vignette with ID '{vignette_id}' not found.")
    return vignette_path


# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Spatial Vignettes Server is running."}


# ---
# API: Upload Raw Vignette Data
# ---
@app.post(
    "/vignettes/",
    response_model=schemas.VignetteCreatedResponse,
    status_code=201,
    tags=["Vignette Lifecycle"]
)
async def create_vignette(
    rgb_image: UploadFile = File(..., description="The captured RGB image (e.g., in PNG format)."),
    depth_data: UploadFile = File(..., description="The captured depth data as a binary blob."),
    confidence_data: Optional[UploadFile] = File(..., description="The confidence data."),
    metadata: str = Body(..., description="A JSON string containing the VignetteMetadata.")
):
    """
    Receives all the raw data for a new vignette from the capture app.
    It creates a unique ID, saves the data, and returns the ID to the app.

    How to call from Swift:
    - Create a multipart/form-data request.
    - Add the image data as a file part named 'rgb_image'.
    - Add the depth data as a file part named 'depth_data'.
    - Serialize metadata object to a JSON string and add it as a form field named 'metadata'.
    """
    vignette_id = str(uuid.uuid4())
    vignette_path = DATA_DIR / vignette_id

    try:
        # Save the uploaded files
        vignette_path.mkdir()
        logging.info(f"Creating new vignette with ID: {vignette_id}")

        with open(vignette_path / "rgb.png", "wb") as f:
            f.write(await rgb_image.read())
        with open(vignette_path / "depth.bin", "wb") as f:
            f.write(await depth_data.read())

        if confidence_data:
            with open(vignette_path / "confidence.bin", "wb") as f:
                f.write(await confidence_data.read())

        # Parse the JSON string from the metadata field, validate it, and save it
        parsed_metadata = json.loads(metadata)
        schemas.VignetteMetadata(**parsed_metadata)  #Validates format
        with open(vignette_path / "metadata.json", "w") as f:
            json.dump(parsed_metadata, f, indent=2)

    except Exception as e:
        logging.error(f"Failed to create vignette {vignette_id}: {e}")
        if vignette_path.exists(): shutil.rmtree(vignette_path)
        raise HTTPException(status_code=500, detail=f"Failed to save vignette data: {str(e)}")

    return {"vignette_id": vignette_id, "status": "Vignette created successfully"}


# ---
# API: On-Demand Segmentation
# ---
@app.post(
    "/vignettes/{vignette_id}/segmentation/logits/",
    response_model=schemas.LogitsResponse,
    tags=["Segmentation"]
)
def get_segmentation_logits(vignette_path: Path = Depends(get_vignette_path)):
    """
    Triggers the AI model to generate raw segmentation logits. This is the slow,
    heavy-lifting part of the process. It should be called once after upload.
    """
    rgb_path = vignette_path / "rgb.png"
    metadata_path = vignette_path / "metadata.json"

    if not all([vignette_path.exists(), rgb_path.exists(), metadata_path.exists()]):
        raise HTTPException(status_code=404, detail="Vignette data not found.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    uv_coords = metadata.get("subject_uv")
    if not uv_coords:
        raise HTTPException(status_code=400, detail="`subject_uv` not found in metadata.")

    try:
        results_path = vignette_path / "results"
        logits_path = segmentation.generate_sam_logits(
            rgb_image_path=rgb_path,
            uv_coords=uv_coords,
            output_path=results_path
        )

        # 1. Load the raw logits from the .npy file.
        logits_array = np.load(logits_path)
        # 2. Convert the NumPy array to raw bytes.
        logits_bytes = logits_array.tobytes()
        # 3. Encode the raw bytes as a Base64 string for safe JSON transfer.
        logits_base64 = base64.b64encode(logits_bytes).decode('utf-8')

        logging.info(f"Successfully generated and encoded logits for vignette: {vignette_path.name}")
        return {
            "status": "success",
            "detail": "Logits generated and returned successfully.",
            "logits_base64": logits_base64,
            "mask_shape": list(logits_array.shape) # [height, width]
        }
    except Exception as e:
        logging.error(f"Error during segmentation for {vignette_path.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {e}")


@app.post(
    "/vignettes/{vignette_id}/segmentation/mask/",
    response_model=schemas.StatusResponse,
    tags=["Segmentation"]
)
def get_segmentation_mask(body: schemas.SegmentationThreshold, vignette_path: Path = Depends(get_vignette_path)):
    """
    Applies a threshold to the pre-computed logits to generate a final binary mask.
    This is very fast and can be called repeatedly from the app (e.g., from a slider).
    """
    results_path = vignette_path / "results"
    logits_path = results_path / "mask_logits.npy"

    if not logits_path.exists():
        raise HTTPException(status_code=404, detail="Logits not found. Run the logits endpoint first.")

    try:
        mask_path = segmentation.apply_threshold_to_logits(
            logits_path=logits_path,
            threshold=body.threshold,
            output_path=results_path
        )
        logging.info(f"Successfully generated mask for vignette: {vignette_path.name} with threshold {body.threshold}")
        return {
            "status": "Mask generated successfully",
            "mask_path": str(mask_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying threshold: {e}")

