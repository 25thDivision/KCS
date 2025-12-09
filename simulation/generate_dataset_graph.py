import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generators.color_code import create_color_code_circuit, generate_dataset
from common.mapper_graph import SyndromeGraphMapper

# ==============================================================================
# [설정] 대용량 데이터 생성 (Train + Test)
# ==============================================================================
TRAIN_SAMPLES = {
    3: 10000000, 
    5: 1000000, 
    7: 1000000
}

TEST_SAMPLES = {
    3: 100000,
    5: 100000,
    7: 100000
}

BATCH_SIZE = 100000 

NOISE_RATES = [0.005, 0.01, 0.05]
ERROR_TYPES = ["X", "Z"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/color_code/graph")

def generate_and_save(mapper, d, p, err_type, total_samples, file_prefix):
    all_features = []
    all_labels = []
    
    num_batches = int(np.ceil(total_samples / BATCH_SIZE))
    desc = f"       [{file_prefix.upper()}] ({err_type}, p={p})"
    
    for b in tqdm(range(num_batches), desc=desc):
        current_batch_size = min(BATCH_SIZE, total_samples - b * BATCH_SIZE)
        
        raw_detectors, physical_errors = generate_dataset(
            d, d, p, current_batch_size, error_type=err_type
        )
        
        features = mapper.map_to_node_features(raw_detectors)
        
        all_features.append(features)
        all_labels.append(physical_errors)
    
    full_features = np.concatenate(all_features, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    
    file_name = f"{file_prefix}_d{d}_p{p}_{err_type}.npz"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    np.savez_compressed(
        file_path,
        features=full_features,
        labels=full_labels
    )
    print(f"       Saved: {file_name} (Shape: {full_features.shape})")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"=== Generating Graph Datasets (Train & Test) ===")
    
    for d, train_count in TRAIN_SAMPLES.items():
        test_count = TEST_SAMPLES[d]
        print(f"\n>>> Processing Distance d={d} (Train: {train_count}, Test: {test_count})")
        
        circuit = create_color_code_circuit(d, d, 0.001)
        mapper = SyndromeGraphMapper(circuit)
        
        # 엣지 정보 저장 (한 번만)
        edge_path = os.path.join(OUTPUT_DIR, f"edges_d{d}.npy")
        np.save(edge_path, mapper.get_edges())
        print(f"    - Saved Edges: {edge_path}")

        for p in NOISE_RATES:
            for err_type in ERROR_TYPES:
                print(f"    -> Processing {err_type}-Error (p={p})...")
                
                # 1. Train 생성
                # generate_and_save(mapper, d, p, err_type, train_count, "train")
                
                # 2. Test 생성 (필수!)
                generate_and_save(mapper, d, p, err_type, test_count, "test")

    print("\n=== All Graph Datasets Generated Successfully! ===")

if __name__ == "__main__":
    main()