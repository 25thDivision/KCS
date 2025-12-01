import os
import sys
import numpy as np
import stim
from tqdm import tqdm

# [수정] 현재 폴더(simulation)를 경로에 추가해서 하위 모듈을 찾게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [수정] 바뀐 경로에 맞춰 Import
from generators.color_code import create_color_code_circuit, generate_dataset
from common.mapper_graph import SyndromeGraphMapper

# ==============================================================================
# 설정 (Configuration)
# ==============================================================================
DISTANCE = 5
ROUNDS = 5
NOISE_RATE_TRAIN = 0.05  # 학습용 노이즈 (조금 어렵게 설정하는 것이 일반적)
NUM_TRAIN = 100000       # 학습 데이터 개수 (예: 10만 개)
NUM_TEST = 10000         # 테스트 데이터 개수 (예: 1만 개)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/color_code/graph")     # 저장할 폴더 경로

def save_dataset(mode, num_samples, noise_rate, mapper):
    print(f"\n>>> Generating {mode} set ({num_samples} samples, p={noise_rate})...")
    
    # 1. Raw Data 생성 (Stim)
    # 메모리 부족 방지를 위해 1만 개씩 끊어서 처리할 수도 있지만, 
    # 일단 10만 개 정도는 메모리에 올라가므로 한 번에 합니다.
    raw_detectors, physical_errors = generate_dataset(DISTANCE, ROUNDS, noise_rate, num_samples)
    
    # 2. Graph Feature 변환
    print(f"    - Converting to Graph Features...")
    node_features = mapper.map_to_node_features(raw_detectors)
    
    # 3. 저장
    file_path = os.path.join(OUTPUT_DIR, f"{mode}_d{DISTANCE}_p{noise_rate}.npz")
    
    # 압축 저장 (.npz)
    np.savez_compressed(
        file_path,
        features=node_features,  # 입력 (X)
        labels=physical_errors   # 정답 (Y)
    )
    print(f"    - Saved to {file_path}")
    print(f"      X shape: {node_features.shape}, Y shape: {physical_errors.shape}")

def main():
    # 저장 폴더 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"=== Generating Dataset for Graph Transformer (d={DISTANCE}) ===")
    
    # 1. Mapper 초기화 (회로 구조 분석용, 한 번만 하면 됨)
    print(">>> Initializing Circuit and Mapper...")
    # 회로 구조용 (노이즈는 상관없음)
    circuit = create_color_code_circuit(DISTANCE, ROUNDS, 0.001)
    mapper = SyndromeGraphMapper(circuit)
    
    # 2. 엣지 정보(Adjacency Matrix) 저장 
    # (그래프 구조는 모든 데이터에서 똑같으므로 따로 저장합니다)
    edge_index = mapper.get_edges()
    edge_path = os.path.join(OUTPUT_DIR, f"edges_d{DISTANCE}.npy")
    np.save(edge_path, edge_index)
    print(f"    - Graph Structure (Edges) saved to {edge_path}")
    
    # 3. 학습 데이터 (Train) 생성
    save_dataset("train", NUM_TRAIN, NOISE_RATE_TRAIN, mapper)
    
    # 4. 테스트 데이터 (Test) 생성
    # 벤치마킹을 위해 여러 노이즈 레벨로 만들 수도 있지만, 일단 하나만 만듭니다.
    save_dataset("test", NUM_TEST, NOISE_RATE_TRAIN, mapper)
    
    print("\n=== All Datasets Generated Successfully! ===")

if __name__ == "__main__":
    main()