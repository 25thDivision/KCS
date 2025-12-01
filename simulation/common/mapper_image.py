import stim
import numpy as np
from typing import Dict, List

class SyndromeImageMapper:
    """
    Handles the mapping from 1D Stim detector indices to 2D grid images.
    This is essential for using CNN-based models (e.g., U-Net) with topological codes.
    """

    def __init__(self, circuit: stim.Circuit):
        """
        Initializes the mapper by parsing detector coordinates from the circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit containing detector coordinate data.
        """
        # 1. Extract detector coordinates from Stim
        # Format: {detector_index: [x, y, ...]}
        coords_dict: Dict[int, List[float]] = circuit.get_detector_coordinates()
        
        # 2. Separate indices and coordinates
        self.indices = np.array(list(coords_dict.keys()))
        coords_values = np.array(list(coords_dict.values()))
        
        # We assume 2D coordinates. 
        # If the circuit is 3D (space-time), we typically project to 2D or use the first two dimensions.
        self.xs = coords_values[:, 0]
        self.ys = coords_values[:, 1]
        
        # 3. Normalize coordinates to 0-based integer indices
        # This shifts the grid so the top-leftmost detector is at (0, 0)
        self.min_x = np.min(self.xs)
        self.min_y = np.min(self.ys)
        
        # Calculate the spatial dimensions of the grid
        self.width = int(np.max(self.xs) - self.min_x) + 1
        self.height = int(np.max(self.ys) - self.min_y) + 1
        
        # 4. Pre-calculate the target (row, col) for each detector index
        # These arrays act as a lookup table for fast mapping
        self.mapped_rows = (self.ys - self.min_y).astype(int)
        self.mapped_cols = (self.xs - self.min_x).astype(int)
        
        print(f"[SyndromeImageMapper] Initialized. Grid Size: {self.height}x{self.width}")

    def map_to_images(self, detector_data: np.ndarray) -> np.ndarray:
        """
        Converts a batch of 1D detector data into 2D syndrome images.

        Args:
            detector_data (np.ndarray): 1D boolean array of shape [shots, num_detectors].

        Returns:
            np.ndarray: 2D image array of shape [shots, 1, height, width].
                        Using 'float32' is standard for Neural Network inputs.
        """
        num_shots = detector_data.shape[0]
        
        # Initialize empty images with shape (N, C, H, W)
        # C=1 because syndrome data is grayscale (single channel)
        images = np.zeros((num_shots, 1, self.height, self.width), dtype=np.float32)
        
        # Vectorized Mapping
        # Iterate over each valid detector and place its values onto the grid across all shots
        for i, detector_idx in enumerate(self.indices):
            # Check boundary to prevent errors if data shape doesn't match circuit
            if detector_idx < detector_data.shape[1]:
                # Extract the column for this detector across all shots
                column_data = detector_data[:, detector_idx]
                
                # Assign to the corresponding pixel (row, col)
                images[:, 0, self.mapped_rows[i], self.mapped_cols[i]] = column_data
                
        return images