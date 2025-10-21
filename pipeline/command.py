# file: commands.py

from typing import List, Literal, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

# --- Part A: Raw User Input Payloads ---
# These models represent the geometric data coming from the client app.

class TapInputPayload(BaseModel):
    """Payload for a single tap action."""
    uv_coord: Tuple[float, float] = Field(..., description="The (u,v) coordinate of the tap, normalized from 0.0 to 1.0.")

class StrokeInputPayload(BaseModel):
    """Payload for a user-drawn line or curve."""
    uv_coords: List[Tuple[float, float]] = Field(..., description="An ordered list of (u,v) coordinates representing the stroke.")

class LassoInputPayload(BaseModel):
    """Payload for a closed, user-drawn region."""
    uv_coords: List[Tuple[float, float]] = Field(..., description="An ordered list of (u,v) coordinates forming a closed polygon.")


# --- Part B: Specific Command Models ---
# These models combine a payload with a specific design intent and parameters.

class BaseCommand(BaseModel):
    """An abstract base model for all commands."""
    command_name: str
    input_type: Literal["tap", "stroke", "lasso"]


class SelectRegionCommand(BaseCommand):
    """Command to select points within a region."""
    command_name: Literal["select_region"] = "select_region"
    input_type: Literal["lasso"] = "lasso"
    payload: LassoInputPayload
    
    class Params(BaseModel):
        new_attribute_name: str = Field("user_selection", description="The name for the new boolean attribute marking selected points.")
    params: Params = Params()


class DefineAxisCommand(BaseCommand):
    """Command to define a structural axis from a user stroke."""
    command_name: Literal["define_axis"] = "define_axis"
    input_type: Literal["stroke"] = "stroke"
    payload: StrokeInputPayload
    
    class Params(BaseModel):
        abstraction_name: str = Field("user_axis", description="The name to store this axis under in the vignette's abstractions.")
        force_pca_alignment: bool = Field(True, description="If True, re-runs PCA on nearby points, guided by this line.")
    params: Params = Params()


class GenerateSurfaceCommand(BaseCommand):
    """Command to generate a new surface by sweeping a stroke."""
    command_name: Literal["generate_surface"] = "generate_surface"
    input_type: Literal["stroke"] = "stroke"
    payload: StrokeInputPayload

    class Params(BaseModel):
        abstraction_name: str = Field("user_surface", description="The name for the generated mesh abstraction.")
        flow_attribute: str = Field("flow_vectors", description="The per-point attribute to use for the sweep direction.")
        sweep_width: float = Field(0.1, description="The width of the swept surface in 3D units.")
        sweep_steps: int = Field(20, description="The number of steps to take along the flow vectors.")
    params: Params = Params()