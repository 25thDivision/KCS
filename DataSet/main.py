import matplotlib.pyplot as plt
import numpy as np
import torch  # Imported to ensure compatibility checks later

# Import custom modules for data generation and mapping
from syndrome_gen import create_color_code_circuit, generate_dataset
from mapper_image import SyndromeImageMapper
from mapper_graph import SyndromeGraphMapper

# ==============================================================================
# Configuration Parameters
# ==============================================================================
# Distance (d): The size of the code. Determines the number of qubits and detectors.
# A larger distance means better error protection but a larger simulation overhead.
DISTANCE = 5

# Rounds (T): The number of repeated syndrome measurement cycles.
# This adds the 'time' dimension to handle measurement errors.
ROUNDS = 5

# Physical Error Rate (p): The probability of an error occurring at each operation.
# e.g., 0.01 means a 1% chance of depolarization error per step.
NOISE_RATE = 0.20

# Number of Shots: How many independent experiments (samples) to simulate.
# We use a small number here for verification purposes.
NUM_SHOTS = 3

def inspect_shot_details(shot_idx, raw_detectors, mapper):
    """
    Specific utility to see which detectors fired and where they are on the grid.
    """
    # 1. 해당 샷(Shot)의 데이터 가져오기
    shot_data = raw_detectors[shot_idx]

    # 2. 값이 1인(에러가 발생한) 탐지기의 인덱스 찾기 (1D)
    # np.flatnonzero: 0이 아닌 값의 인덱스를 싹 긁어옵니다.
    fired_indices = np.flatnonzero(shot_data)

    print(f"\n=== 🔍 상세 분석 (Shot #{shot_idx}) ===")
    print(f"총 {len(fired_indices)}개의 탐지기가 켜졌습니다.")
    print("-" * 50)
    print(f"{'Detector ID (1D)':<20} | {'Grid Coord (Y, X)':<20}")
    print("-" * 50)

    # 3. 각 인덱스에 해당하는 2D 좌표 찾기
    for idx in fired_indices:
        # mapper.indices 배열에서 해당 탐지기 ID가 몇 번째에 있는지 찾음
        lookup_loc = np.where(mapper.indices == idx)[0]

        if len(lookup_loc) > 0:
            i = lookup_loc[0] # 매퍼 내부에서의 순번
            r = mapper.mapped_rows[i] # Y 좌표 (Row)
            c = mapper.mapped_cols[i] # X 좌표 (Column)
            print(f"Detector {idx:<11} | (y={r}, x={c})")
        else:
            print(f"Detector {idx:<11} | 매핑 정보 없음 (누락됨?)")
    print("-" * 50)
    
    

def main():
    """
    Main execution function to verify the entire pipeline:
    1. Generate Quantum Data (Stim)
    2. Preprocess for CNN (Image Mapping)
    3. Preprocess for GNN (Graph Mapping)
    """
    print(f"=== Project: Color Code Decoding Benchmark (d={DISTANCE}, p={NOISE_RATE}) ===\n")

    # ==========================================================================
    # Step 1: Generate Raw Quantum Data
    # ==========================================================================
    # We create the quantum circuit definition using Stim.
    print(">>> [Step 1] Generating Circuit and Raw Data...")
    circuit = create_color_code_circuit(DISTANCE, ROUNDS, NOISE_RATE)
    
    # Sample syndromes (detectors) and logical flip information (observables).
    # - raw_detectors: The input data (X) for our models.
    # - raw_observables: The target labels (Y) we want to predict.
    raw_detectors, raw_observables = generate_dataset(circuit, NUM_SHOTS)
    
    print(f"    - Raw Detector Data Shape:   {raw_detectors.shape}")
    print(f"      (Format: [Num_Shots, Num_Detectors])")
    print(f"    - Raw Observable Data Shape: {raw_observables.shape}")
    print(f"      (Format: [Num_Shots, Num_Observables])")
    print("    -> Step 1 Complete.\n")

    # ==========================================================================
    # Step 2: Map to 2D Images (for CNN / U-Net)
    # ==========================================================================
    # CNNs require spatial data structure (Grid).
    # The Mapper converts the 1D detector list into a 2D image based on coordinates.
    print(">>> [Step 2] Mapping to 2D Images (for CNN/U-Net)...")
    image_mapper = SyndromeImageMapper(circuit)
    
    # Perform the conversion.
    # Output format is compatible with PyTorch: (Batch, Channel, Height, Width)
    syndrome_images = image_mapper.map_to_images(raw_detectors)
    
    print(f"    - Mapped Image Shape: {syndrome_images.shape}")
    print(f"      (Format: [Batch_Size, Channels, Height, Width])")
    print(f"    - Grid Dimensions:    {image_mapper.height} x {image_mapper.width}")
    
    # Visualization: Plot the first sample to visually verify the mapping.
    # If the mapping is correct, you should see a sparse grid of dots.
    print("    - Visualizing the first sample...")
    plt.figure(figsize=(6, 6))
    
    # Visualize the first shot (index 0) and first channel (index 0).
    # origin='lower' ensures the (0,0) coordinate is at the bottom-left.
    plt.imshow(syndrome_images[0, 0], origin='lower', cmap='Reds', interpolation='nearest')
    plt.title(f"Syndrome Image Visualization (d={DISTANCE})")
    plt.colorbar(label="Syndrome Triggered (1.0) / Quiet (0.0)")
    plt.xlabel("X Coordinate (Grid Index)")
    plt.ylabel("Y Coordinate (Grid Index)")
    plt.show()
    print("    -> Step 2 Complete.\n")

    # ==========================================================================
    # Step 3: Map to Graph Nodes & Edges (for GNN / Graph Transformer)
    # ==========================================================================
    # GNNs require a graph structure: Nodes (features) and Edges (connectivity).
    # The GraphMapper extracts this topology from the circuit's error model (DEM).
    print(">>> [Step 3] Mapping to Graph Nodes & Edges (for GNN)...")
    graph_mapper = SyndromeGraphMapper(circuit)
    
    # 1. Node Features: The status of each detector (0 or 1).
    # Reshaped to [Batch, Num_Nodes, Num_Features].
    node_features = graph_mapper.map_to_node_features(raw_detectors)
    
    # 2. Edges: The connections between detectors.
    # Extracted from physical error mechanisms (e.g., an error triggering two detectors).
    edges = graph_mapper.get_edges()
    
    print(f"    - Node Features Shape: {node_features.shape}")
    print(f"      (Format: [Batch, Num_Nodes, Feature_Dim])")
    print(f"    - Edge Index Shape:    {edges.shape}")
    print(f"      (Format: [2, Num_Edges] - Source/Target pairs)")
    
    print(f"    - Graph Statistics:    {graph_mapper.num_nodes} Nodes, {edges.shape[1]} Edges found.")
    
    # Logic Check: If physical noise > 0, edges must exist in the graph.
    if edges.shape[1] > 0:
        print("    - Graph connectivity check: Passed (Edges exist).")
    else:
        print("    - WARNING: No edges found. This is unexpected for noisy circuits.")
        
    print("    -> Step 3 Complete.\n")
    
    # 0번째 샷에 대해 1D -> 2D 상세 정보 출력
    inspect_shot_details(0, raw_detectors, image_mapper)

    print("=== All Checks Passed Successfully! ===")

if __name__ == "__main__":
    main()