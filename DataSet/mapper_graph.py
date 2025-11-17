import stim
import numpy as np
from typing import Tuple, Set

class SyndromeGraphMapper:
    """
    Handles the mapping from Stim circuit data to Graph structures for GNNs.
    Constructs a graph where:
    - Nodes: Detectors (syndromes).
    - Edges: Connected if a single error mechanism triggers both detectors.
    """

    def __init__(self, circuit: stim.Circuit):
        """
        Initializes the mapper and builds the static graph structure.

        Args:
            circuit (stim.Circuit): The Stim circuit to analyze.
        """
        # 1. Get detector coordinates (Node Positions)
        # Format: {detector_index: [x, y, ...]}
        self.coords_dict = circuit.get_detector_coordinates()
        self.node_indices = np.array(list(self.coords_dict.keys()))
        
        # Sort indices to ensure consistent ordering (0, 1, 2, ...)
        self.node_indices.sort()
        self.num_nodes = len(self.node_indices)
        
        # Create a lookup table for fast index validation
        self.valid_indices_set = set(self.node_indices)

        # 2. Build Graph Edges from Detector Error Model (DEM)
        # This captures the physical dependency between detectors.
        print("[SyndromeGraphMapper] Building graph edges from DEM... (this may take a moment)")
        self.edge_index = self._build_edges_from_dem(circuit)
        
        print(f"[SyndromeGraphMapper] Initialized. Nodes: {self.num_nodes}, Edges: {self.edge_index.shape[1]}")

    def _build_edges_from_dem(self, circuit: stim.Circuit) -> np.ndarray:
        """
        Internal helper to extract edges.
        Two detectors are connected if they are triggered by the same error mechanism.
        
        Returns:
            np.ndarray: Edge list of shape [2, num_edges] (Source, Target).
        """
        dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        edges: Set[Tuple[int, int]] = set()

        # Iterate over all instructions in the flattened DEM
        for instruction in dem.flattened():
            # We only care about 'error' instructions that introduce noise
            if instruction.type == "error":
                # Extract targets that are relative detectors
                dets = [t.val for t in instruction.targets_copy() if t.is_relative_detector_id()]
                
                # If an error triggers 2 or more detectors, they are correlated.
                # We add edges between all pairs of detectors triggered by this error.
                for i in range(len(dets)):
                    for j in range(i + 1, len(dets)):
                        u, v = dets[i], dets[j]
                        # Ensure both nodes exist in our valid set before adding the edge
                        if u in self.valid_indices_set and v in self.valid_indices_set:
                            # Add undirected edge (both directions)
                            edges.add((u, v))
                            edges.add((v, u))

        # Handle case with no edges (e.g., noiseless circuit)
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
            
        return np.array(list(edges)).T.astype(np.int64)

    def map_to_node_features(self, detector_data: np.ndarray) -> np.ndarray:
        """
        Maps raw detector data to node features.

        Args:
            detector_data (np.ndarray): 1D boolean array [shots, num_detectors].

        Returns:
            np.ndarray: Node features [shots, num_nodes, num_features].
                        Here, num_features=1 (the syndrome status).
        """
        shots = detector_data.shape[0]
        
        # Create feature matrix (Batch, Nodes, Features)
        # Feature dimension is 1 because the syndrome is binary (0 or 1)
        node_features = np.zeros((shots, self.num_nodes, 1), dtype=np.float32)
        
        # Assign values
        # We explicitly map columns to ensure alignment with node_indices
        if detector_data.shape[1] >= self.num_nodes:
            node_features[:, :, 0] = detector_data[:, self.node_indices]
            
        return node_features

    def get_edges(self) -> np.ndarray:
        """
        Returns the static edge list (Adjacency).
        
        Returns:
            np.ndarray: [2, num_edges] array.
        """
        return self.edge_index