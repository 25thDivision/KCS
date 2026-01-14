import os
import sys
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed  # [추가] 병렬 처리 라이브러리

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

# [수정] 병렬 처리 청크 사이즈 (48코어 기준 5000~10000 권장)
CHUNK_SIZE = 5000 

# [수정] 사용할 CPU 코어 수 (-1: 모든 코어 사용)
NUM_WORKERS = -1

NOISE_RATES = [0.005, 0.01, 0.05]
ERROR_TYPES = ["X", "Z"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/color_code/graph")

# [추가] 병렬 처리를 위한 단위 작업 함수
def _generate_chunk(d, p, err_type, count):
    """
    작은 단위(Chunk)의 데이터를 생성하여 반환합니다.
    """
    raw, phys = generate_dataset(d, d, p, count, error_type=err_type)
    return raw, phys

def generate_and_save(mapper, d, p, err_type, total_samples, file_prefix):
    num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
    desc = f"       [{file_prefix.upper()}] ({err_type}, p={p})"
    print(f"{desc} -> Generating with {NUM_WORKERS} workers...")
    
    # 1. [핵심] 병렬 실행 (Stim 시뮬레이션 분산 처리)
    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(_generate_chunk)(
            d, p, err_type, 
            min(CHUNK_SIZE, total_samples - i * CHUNK_SIZE)
        )
        for i in tqdm(range(num_chunks), desc="       Processing Chunks")
    )
    
    # 2. 결과 합치기 및 그래프 피처 매핑
    print("       -> Mapping to Graph Features & Merging...")
    
    all_features = []
    all_labels = []
    
    # 생성된 Raw 데이터를 하나씩 꺼내서 그래프 피처로 변환
    # (매핑 작업은 Numpy 연산이라 빠르므로 메인 프로세스에서 수행)
    for raw_detectors, physical_errors in results:
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
        
    print(f"=== Generating Graph Datasets (Parallel Mode: {NUM_WORKERS}) ===")
    
    for d, train_count in TRAIN_SAMPLES.items():
        test_count = TEST_SAMPLES[d]
        print(f"\n>>> Processing Distance d={d} (Train: {train_count}, Test: {test_count})")
        
        # 회로 및 매퍼 초기화 (Distance별로 한 번만 수행)
        circuit = create_color_code_circuit(d, d, 0.001)
        mapper = SyndromeGraphMapper(circuit)
        
        # 엣지 정보 저장 (한 번만)
        edge_path = os.path.join(OUTPUT_DIR, f"edges_d{d}.npy")
        if not os.path.exists(edge_path):
            np.save(edge_path, mapper.get_edges())
            print(f"    - Saved Edges: {edge_path}")
        else:
            print(f"    - Edges already exist: {edge_path}")

        for p in NOISE_RATES:
            for err_type in ERROR_TYPES:
                print(f"    -> Processing {err_type}-Error (p={p})...")
                
                # 1. Train 생성
                generate_and_save(mapper, d, p, err_type, train_count, "train")
                
                # 2. Test 생성
                generate_and_save(mapper, d, p, err_type, test_count, "test")

    print("\n=== All Graph Datasets Generated Successfully! ===")

if __name__ == "__main__":
    main()