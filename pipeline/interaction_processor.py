# file: interaction_processor.py
from vignette_data import ProcessedVignette

def _get_projection_params(vignette: ProcessedVignette) -> Dict[str, Any]:
    """Extracts and calculates all necessary projection data from a vignette."""
    try:
        meta = vignette.metadata["capture_metadata"]
        raw_intrinsics = np.array(meta['camera_intrinsics']['columns']).T
        depth_res = meta['resolution']
        center_offset = np.array(meta['center_offset'])

        # Replicate the scaling logic from your creation script
        # Assuming original resolution might not be in metadata, we'll use depth res
        # for a 1:1 scaling, which is often the case.
        depth_w, depth_h = depth_res[0], depth_res[1]
        
        return {
            "K": raw_intrinsics,
            "offset": center_offset,
            "width": depth_w,
            "height": depth_h
        }
    except KeyError as e:
        raise ValueError(f"Vignette is missing required metadata for projection: {e}")



# Work on a manager class later!
'''
import numpy as np
from typing import Optional, Union
from pathlib import Path

from vignette_data import ProcessedVignette
from command import BaseCommand, SelectRegionCommand, DefineAxisCommand, GenerateSurfaceCommand

# A type alias for all possible command objects
AnyCommand = Union[SelectRegionCommand, DefineAxisCommand, GenerateSurfaceCommand]

class VignetteInteractionProcessor:
    """
    Processes user interaction commands to guide abstraction and generate new forms
    on a ProcessedVignette object.

    This class is stateful and is initialized for a specific interaction session
    with a single vignette and its corresponding capture data.
    """
    def __init__(self, vignette: "ProcessedVignette"):
        """
        Initializes the processor with a vignette and necessary projection info.
        """
        
        self.vignette = vignette
        self.vignette_folder = self.vignette.file_path.parent.parent

        # private attributes
        self._rgb_image = None
        self._depth_map = None
        self._confidence_map = None
        self._mask = None
        self._metadata = self.vignette.metadata

        self._rgb_path = vignette_path / "rgb.png"
        self._depth_path = vignette_path / "depth.bin"
        self._confidence_path = vignette_path / "confidence.bin"
       self._ metadata_path = vignette_path / "metadata.json"
        mask_path = vignette_path / "results" / "mask.png"
        # load these data when needed in custom methods?
        
        
        print(f"Processor initialized for vignette: {self.vignette.file_path}")

    # --- Public Methods ---

    def process_command(self, command: AnyCommand) -> Optional["ProcessedVignette"]:
        """
        Main entry point. Dispatches a command object to the appropriate handler.
        """
        print(f"Received command: '{command.command_name}'")

        if isinstance(command, SelectRegionCommand):
            return self._handle_select_region(command)
        
        elif isinstance(command, DefineAxisCommand):
            return self._handle_define_axis(command)
        
        elif isinstance(command, GenerateSurfaceCommand):
            return self._handle_generate_surface(command)
            
        else:
            raise NotImplementedError(f"No handler implemented for command type: {type(command)}")

    # --- Private Command Handlers ---

    def _handle_select_region(self, command: SelectRegionCommand) -> None:
        """
        Processes the SelectRegionCommand to create a new per-point attribute.
        """
        # --- LOGIC TO IMPLEMENT ---
        # 1. Get uv_coords from command.payload.
        # 2. Project all 3D points from self.vignette.points to 2D using self.pose and self.K.
        # 3. Create a 2D mask from the uv_coords.
        # 4. Create a 1D boolean mask for the points based on which ones fall inside the 2D mask.
        # 5. Use self.vignette.set_attribute() with the name from command.params.new_attribute_name.
        pass

    def _handle_define_axis(self, command: DefineAxisCommand) -> None:
        """
        Processes the DefineAxisCommand to create a new axis abstraction.
        """
        # --- LOGIC TO IMPLEMENT ---
        # 1. Get uv_coords from command.payload.
        # 2. Unproject the 2D stroke into a list of 3D points using self._unproject_uv_to_3d.
        # 3. Fit a line (e.g., via PCA) to these 3D points to get an origin and direction.
        # 4. Optionally, use this line to guide a new PCA on nearby points from the main point cloud.
        # 5. Use self.vignette.add_abstraction() to store the new axis.
        pass

    def _handle_generate_surface(self, command: GenerateSurfaceCommand) -> None:
        """
        Processes the GenerateSurfaceCommand to create and store a new mesh abstraction.
        """
        # --- LOGIC TO IMPLEMENT ---
        # 1. Unproject the user's stroke into a 3D path.
        # 2. For each point on the 3D path, find the nearest point in the main point cloud.
        # 3. Get the flow vector (from command.params.flow_attribute) at that nearest point.
        # 4. Generate a ribbon or mesh by extruding the path along these interpolated flow vectors.
        # 5. Store the resulting mesh (vertices, faces) in a new abstraction using self.vignette.add_abstraction().
        pass

    # --- Private Helper Methods ---

    def _unproject_uv_to_3d(self, uv_coords: List[Tuple[float, float]]) -> np.ndarray:
        """
        Converts a list of normalized (u,v) coordinates to 3D points using the depth map.
        """
        # --- LOGIC TO IMPLEMENT ---
        # This will contain the math for converting 2D+depth to 3D.
        return np.array([])

    def _project_3d_to_uv(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Projects 3D points from the vignette into normalized (u,v) coordinates.
        """
        # --- LOGIC TO IMPLEMENT ---
        # This will contain the math for converting 3D to 2D.
        return np.array([])
'''