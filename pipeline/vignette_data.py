"""
vignette_data.py - Defines a custom, intelligent data structure for processed vignettes.
"""
import numpy as np
import open3d as o3d
from typing import Optional, Dict, Any, List
from pathlib import Path
import matplotlib.cm as cm

class ProcessedVignette:
    """
    A container for a processed spatial vignette.

    Contains the following
    1. Core geometry: points, colors
    2. Per-point attributes
    3. Metadata & abstractions
    """
    def __init__(self,
                 points: np.ndarray,
                 colors: np.ndarray,
                 normals: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 **attributes: np.array
                 ):
        
        # Path
        self.file_path: Optional[Path] = None
        
        # Core geometry
        self.points = points
        self.colors = colors
        self.normals = normals
        self.n_points = len(points)
        
        # Per-point attributes
        self.attributes: Dict[str, np.ndarray] = {}
        for name, values in attributes.items():
            self.set_attribute(name, values, auto_save=False)
        # Non-per-point data like abstraction results
        self.metadata = metadata or {}

    # --- Per-Point Attribute ---

    def set_attribute(self, name: str, values: np.ndarray, auto_save: bool = True):
        """
        Sets a custom per-point attribute and auto-saves the vignette.
        """
        if not isinstance(values, np.ndarray) or values.shape[0] != self.n_points:
            raise ValueError(
                f"Attribute '{name}' size mismatch: "
                f"Expected {self.n_points} values, but got {values.shape[0]}."
            )
        
        self.attributes[name] = values
        print(f"Set/updated per-point attribute: '{name}'.")
        
        if auto_save and self.file_path:
            self.save()

    def get_attribute(self, name: str) -> Optional[np.ndarray]:
        """Gets a custom per-point attribute by name."""
        return self.attributes.get(name)
    
    def __getattr__(self, name: str) -> Any:
        """
        Enables convenient "dot-notation" access to per-point attributes.
        """
        if name in self.attributes:
            return self.attributes[name]
        # Not found
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'.")

    # --- Abstraction ---

    def add_abstraction(self, shape_type: str, shape_data: Dict[str, Any], auto_save: bool = True):
        """
        Adds a high-level geometric abstraction (like a plane or axis) to the metadata.

        Args:
            shape_type: The category of shape (e.g., 'planes', 'axes', 'meshes').
            shape_data: A dictionary containing the shape's parameters.
            auto_save: If True, saves the vignette after modification.
        """
        # Ensure the top-level 'abstractions' key exists in our metadata.
        if 'abstractions' not in self.metadata:
            self.metadata['abstractions'] = {}
        
        # Ensure the list for this specific shape type exists.
        if shape_type not in self.metadata['abstractions']:
            self.metadata['abstractions'][shape_type] = []
            
        self.metadata['abstractions'][shape_type].append(shape_data)
        print(f"Added new abstraction of type '{shape_type}'.")
        
        if auto_save and self.file_path:
            self.save()

    def get_abstractions(self, shape_type: Optional[str] = None) -> Optional[Any]:
        """
        Retrieves stored geometric abstractions.

        Args:
            shape_type: The type of shape to retrieve (e.g., 'planes'). If None,
                        returns the entire abstractions dictionary.
        """
        abstractions = self.metadata.get('abstractions')
        if abstractions is None:
            return None
        
        if shape_type:
            return abstractions.get(shape_type)
        else:
            return abstractions

    def clear_abstractions(self, shape_type: Optional[str] = None, auto_save: bool = True):
        """
        Removes stored geometric abstractions to allow for re-computation.
        """
        if 'abstractions' not in self.metadata:
            return

        if shape_type:
            if shape_type in self.metadata['abstractions']:
                count = len(self.metadata['abstractions'][shape_type])
                del self.metadata['abstractions'][shape_type]
                print(f"Cleared {count} abstractions of type '{shape_type}'.")
        else:
            self.metadata['abstractions'] = {}
            print("Cleared all abstractions.")
            
        if auto_save and self.file_path:
            self.save()

    def save(self, file_path: Optional[str] = None):
        """
        Saves all vignette data to a single compressed .npz file.
        If no path is provided, it overwrites its original file.
        """
        path_to_save = Path(file_path) if file_path else self.file_path
        if not path_to_save:
            raise ValueError("No file path specified for saving. Provide a path or load the vignette from a file first.")

        np.savez_compressed(
            path_to_save,
            # Core data
            points=self.points,
            colors=self.colors,
            normals=self.normals,
            # Metadata: Wrap the dict in a 0-D array to save it in the .npz format.
            metadata=np.array(self.metadata),
            # All flexible per-point attributes are automatically included
            **self.attributes 
        )

        self.file_path = path_to_save # Update the path
        print(f"Saved processed vignette to: {self.file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'ProcessedVignette':
        """
        Factory method to load a vignette from a .npz file.
        It automatically discovers and loads all core data and custom attributes.
        """
        # Using a `with` block ensures the file is properly closed.
        # `allow_pickle=True` is required to load the 'metadata' object array.
        with np.load(file_path, allow_pickle=True) as data:
            # Define the keys for core data that we'll handle explicitly.
            core_keys = ['points', 'colors', 'normals', 'metadata']
            
            # Load core data
            points = data['points']
            colors = data['colors']
            
            # Handle the case where 'normals' was saved as None, which numpy
            # stores as an array containing the None object.
            normals_data = data.get('normals')
            if normals_data is not None and normals_data.ndim == 0 and normals_data.item() is None:
                normals = None
            else:
                normals = normals_data
            
            # Load metadata: Unwrap the dictionary from its 0-D array container.
            metadata_item = data.get('metadata')
            metadata = metadata_item.item() if metadata_item is not None else {}
            
            # Auto-discovery of custom attributes: Any key in the file that isn't
            # a core key is treated as a custom per-point attribute.
            attributes = {
                key: data[key] for key in data.keys() if key not in core_keys
            }

        # Create the new instance, passing the discovered attributes as keyword arguments.
        instance = cls(
            points=points,
            colors=colors,
            normals=normals,
            metadata=metadata,
            **attributes
        )
        
        instance.file_path = Path(file_path)
        print(f"Saved processed vignette from: {file_path}")
        return instance


    # --- Utility and Visualization ---

    def to_open3d(self, color_mode: str = "rgb") -> o3d.geometry.PointCloud:
        """
        Converts the stored data into an Open3D PointCloud object for visualization.
        This method can color the point cloud based on any stored per-point attribute.
        
        Args:
            color_mode: The attribute to use for coloring. Defaults to 'rgb'.
                        Can be 'rgb' or the name of any custom attribute
                        (e.g., 'confidence', 'plane_id').
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)

        print(f"Generating Open3D point cloud with '{color_mode}' colors...")
        
        if color_mode == "rgb":
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        else:
            values = self.get_attribute(color_mode)
            if values is not None:
                # Intelligently apply colormaps based on data type.
                if np.issubdtype(values.dtype, np.integer): # Categorical data (e.g., IDs)
                    unique_values = np.unique(values)
                    # Use a colormap with distinct colors for different categories.
                    colormap = cm.get_cmap("tab20", len(unique_values)) 
                    color_map = {val: colormap(i)[:3] for i, val in enumerate(unique_values)}
                    # Often, a label of 0 or -1 means "no category", so we color it gray.
                    if 0 in color_map: color_map[0] = [0.5, 0.5, 0.5]
                    if -1 in color_map: color_map[-1] = [0.5, 0.5, 0.5]
                    new_colors = np.array([color_map.get(val, [0,0,0]) for val in values])
                else: # Continuous data (e.g., confidence, curvature)
                    min_val, max_val = values.min(), values.max()
                    if min_val == max_val: norm_values = np.zeros_like(values)
                    else: norm_values = (values - min_val) / (max_val - min_val)
                    colormap = cm.get_cmap("viridis")
                    new_colors = colormap(norm_values)[:, :3]
                
                pcd.colors = o3d.utility.Vector3dVector(new_colors)
            else:
                print(f"Warning: Attribute '{color_mode}' not found. Defaulting to gray.")
                pcd.paint_uniform_color([0.5, 0.5, 0.5])
                
        return pcd