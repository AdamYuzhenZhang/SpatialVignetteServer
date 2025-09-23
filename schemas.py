"""
schemas.py - Pydantic Models

This file defines the data structures for API requests and responses.
By keeping them here, we separate the data contract from the API logic.
"""
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Request Models ---

class CameraIntrinsics(BaseModel):
    """Defines the structure for the 3x3 camera intrinsics matrix."""
    columns: List[List[float]]

class VignetteMetadata(BaseModel):
    """
    Defines the structure for the metadata JSON object. This should perfectly
    match the data you serialize and send from your Swift `SpatialVignette` class.
    """
    resolution: List[int] = Field(..., description="[width, height] of the image.")
    camera_intrinsics: CameraIntrinsics
    subject_uv: List[float] = Field(..., description="[u, v] coordinates of the user's tap.")
    # You can add other fields here as needed (e.g., gps, deviceModel, cameraExtrinsics)

class SegmentationThreshold(BaseModel):
    """Defines the structure for the interactive thresholding request body."""
    threshold: float = Field(0.0, description="Value from the app's slider.")

# --- Response Models ---
# Defining response models makes your API more predictable and better documented.

class VignetteCreatedResponse(BaseModel):
    vignette_id: str
    status: str

class LogitsResponse(BaseModel):
    """The response payload when logits are generated."""
    status: str
    detail: str
    logits_base64: str = Field(..., description="The raw logits array, encoded as a Base64 string.")
    mask_shape: List[int] = Field(..., description="The shape of the mask [height, width].")

class StatusResponse(BaseModel):
    status: str
    detail: Optional[str] = None # An optional field for more information