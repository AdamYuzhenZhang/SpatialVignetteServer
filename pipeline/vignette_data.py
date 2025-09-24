"""
vignette_data.py - Defines a custom, intelligent data structure for processed vignettes.
"""
import numpy as np
import open3d as o3d
from typing import Optional
from pathlib import Path

# We need matplotlib for its powerful colormapping capabilities.
import matplotlib.cm as cm

class ProcessedVignette:
    """
    An intelligent container for a processed spatial vignette.

    This class bundles the core geometry (points, colors, normals) with any
    number of custom per-point attributes. It handles its own saving and loading
    and provides flexible visualization options.
    """
    def __init__(self,
                 points: np.ndarray,
                 colors: np.ndarray,
                 normals: Optional[np.ndarray] = None,
                 confidence: Optional[np.ndarray] = None):
        
        # This path is crucial for the auto-save feature.
        self.file_path: Optional[Path] = None
        
        # Core geometry
        self.points = points
        self.colors = colors
        self.normals = normals
        
        # Custom per-point attributes
        self.confidence = confidence

    def set_attribute(self, name: str, values: np.ndarray):
        """
        Sets a custom per-point attribute and auto-saves the vignette.

        Args:
            name: The name of the new attribute (e.g., "importance").
            values: A NumPy array of values, one for each point.
        """
        if len(values) != len(self.points):
            raise ValueError(f"Attribute size mismatch: Expected {len(self.points)} values, but got {len(values)}.")
        
        # `setattr` dynamically adds the new array as a property of the class instance.
        setattr(self, name, values)
        print(f"Set custom attribute '{name}'.")
        
        # Trigger the auto-save feature.
        self.save()

    def get_attribute(self, name: str) -> Optional[np.ndarray]:
        """Gets a custom per-point attribute by name."""
        return getattr(self, name, None)

    def save(self, file_path: Optional[str] = None):
        """
        Saves all vignette data to a single compressed .npz file.
        If no path is provided, it overwrites its original file.
        """
        path_to_save = Path(file_path) if file_path else self.file_path
        if not path_to_save:
            raise ValueError("No file path specified for saving. Provide a path or load the vignette from a file first.")

        # Dynamically create a dictionary of all NumPy array attributes to save.
        data_dict = {
            attr: value for attr, value in self.__dict__.items()
            if isinstance(value, np.ndarray)
        }
            
        np.savez_compressed(path_to_save, **data_dict)
        self.file_path = path_to_save # Update the path in case it was a new save
        print(f"Saved processed vignette to: {self.file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'ProcessedVignette':
        """Loads a vignette from a .npz file."""
        data = np.load(file_path)
        
        # Create the instance with the required data
        instance = cls(
            points=data['points'],
            colors=data['colors'],
            normals=data.get('normals'),
            confidence=data.get('confidence')
        )
        
        # Dynamically add any other custom attributes found in the file
        for key in data.keys():
            if not hasattr(instance, key):
                setattr(instance, key, data[key])

        instance.file_path = Path(file_path) # Store the path for auto-saving
        print(f"Loaded processed vignette from: {file_path}")
        return instance

    def to_open3d(self, color_mode: str = "rgb") -> o3d.geometry.PointCloud:
        """
        Converts the stored data into an Open3D PointCloud object with flexible coloring.

        Args:
            color_mode (str): The attribute to use for coloring.
                              Can be "rgb", "confidence", or any other custom
                              attribute you've set (e.g., "importance").
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)

        print(f"Generating Open3D point cloud with '{color_mode}' colors...")
        
        if color_mode == "rgb":
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        else:
            # Try to get the custom attribute data
            values = self.get_attribute(color_mode)
            if values is not None:
                # Normalize the values to the [0, 1] range for the colormap
                min_val, max_val = values.min(), values.max()
                if min_val == max_val: # Avoid division by zero
                    norm_values = np.zeros_like(values)
                else:
                    norm_values = (values - min_val) / (max_val - min_val)
                
                # Use a matplotlib colormap to convert the normalized values to RGB
                # `viridis` is a good default perceptually-uniform colormap.
                colormap = cm.get_cmap("viridis")
                new_colors = colormap(norm_values)[:, :3] # Discard the alpha channel
                pcd.colors = o3d.utility.Vector3dVector(new_colors)
            else:
                # If the attribute doesn't exist, default to a gray color
                print(f"Warning: Attribute '{color_mode}' not found. Defaulting to gray.")
                pcd.paint_uniform_color([0.5, 0.5, 0.5])
                
        return pcd