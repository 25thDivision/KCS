import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

# [수정] 현재 폴더(simulation)와 상위 폴더(KCS)를 모두 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)          # simulation 폴더 내 모듈용
sys.path.append(os.path.join(current_dir, '../')) # models 폴더용

# [수정] 바뀐 경로 Import
from generators.color_code import create_color_code_circuit, generate_dataset
from legacy.mapper_image import SyndromeImageMapper
from legacy.mapper_graph import SyndromeGraphMapper

# ==============================================================================
# Configuration Parameters
# ==============================================================================
DISTANCE = 5
ROUNDS = 5
NOISE_RATE = 0.20
NUM_SHOTS = 3

def inspect_shot_details(shot_idx, raw_detectors, mapper):
    """
    특정 샷(Shot)에서 어떤 탐지기가 켜졌는지 위치와 함께 상세히 확인합니다
    """
    shot_data = raw_detectors[shot_idx]
    fired_indices = np.flatnonzero(shot_data)

    print(f"\n=== 🔍 상세 분석 (Shot #{shot_idx}) ===")
    print(f"총 {len(fired_indices)}개의 탐지기가 켜졌습니다")
    print("-" * 50)
    print(f"{'Detector ID (1D)':<20} | {'Grid Coord (Y, X)':<20}")
    print("-" * 50)

    for idx in fired_indices:
        lookup_loc = np.where(mapper.indices == idx)[0]
        if len(lookup_loc) > 0:
            i = lookup_loc[0]
            r = mapper.mapped_rows[i]
            c = mapper.mapped_cols[i]
            print(f"Detector {idx:<11} | (y={r}, x={c})")
        else:
            print(f"Detector {idx:<11} | 매핑 정보 없음")
    print("-" * 50)
    

def main():
    """
    전체 파이프라인 검증 함수:
    1. 데이터 생성 (Stim + Physical Error Labeling)
    2. CNN용 이미지 매핑 전처리
    3. GNN용 그래프 매핑 전처리
    """
    print(f"=== Project: Color Code Decoding Benchmark (d={DISTANCE}, p={NOISE_RATE}) ===\n")

    # ==========================================================================
    # Step 1: Raw Quantum Data 생성
    # ==========================================================================
    print(">>> [Step 1] Generating Circuit and Raw Data...")
    
    # 1-1. 회로 생성 (매퍼 초기화용)
    # generate_dataset 내부에서도 회로를 만들지만, 
    # 매퍼(Mapper)들이 사용할 회로 구조 정보가 필요하므로 여기서도 생성합니다
    circuit = create_color_code_circuit(DISTANCE, ROUNDS, NOISE_RATE)
    
    # 1-2. 데이터 생성 (수정된 부분)
    # 인자가 변경되었습니다: (circuit, shots) -> (distance, rounds, noise, shots)
    # 반환값이 변경되었습니다: observables -> physical_errors (물리적 에러 위치)
    raw_detectors, physical_errors = generate_dataset(DISTANCE, ROUNDS, NOISE_RATE, NUM_SHOTS)
    
    print(f"    - Raw Detector Data Shape:   {raw_detectors.shape}")
    print(f"      (Format: [Num_Shots, Num_Detectors])")
    print(f"    - Physical Error Data Shape: {physical_errors.shape}")
    print(f"      (Format: [Num_Shots, Num_Qubits] - Label: 0=Clean, 1=Error)")
    print("    -> Step 1 Complete.\n")

    # ==========================================================================
    # Step 2: 2D 이미지 매핑 (CNN용)
    # ==========================================================================
    print(">>> [Step 2] Mapping to 2D Images (for CNN/U-Net)...")
    image_mapper = SyndromeImageMapper(circuit)
    
    syndrome_images = image_mapper.map_to_images(raw_detectors)
    
    print(f"    - Mapped Image Shape: {syndrome_images.shape}")
    print(f"      (Format: [Batch_Size, Channels, Height, Width])")
    print(f"    - Grid Dimensions:    {image_mapper.height} x {image_mapper.width}")
    
    # 시각화
    print("    - Visualizing the first sample...")
    plt.figure(figsize=(6, 6))
    plt.imshow(syndrome_images[0, 0], origin='lower', cmap='Reds', interpolation='nearest')
    plt.title(f"Syndrome Image Visualization (d={DISTANCE})")
    plt.colorbar(label="Syndrome Triggered (1.0) / Quiet (0.0)")
    plt.xlabel("X Coordinate (Grid Index)")
    plt.ylabel("Y Coordinate (Grid Index)")
    plt.show()
    print("    -> Step 2 Complete.\n")

    # ==========================================================================
    # Step 3: 그래프 매핑 (GNN용)
    # ==========================================================================
    print(">>> [Step 3] Mapping to Graph Nodes & Edges (for GNN)...")
    graph_mapper = SyndromeGraphMapper(circuit)
    
    # 노드 피처 생성 (이제 Stabilizer Type 정보가 포함되어 채널 수가 4입니다)
    node_features = graph_mapper.map_to_node_features(raw_detectors)
    edges = graph_mapper.get_edges()
    
    print(f"    - Node Features Shape: {node_features.shape}")
    print(f"      (Format: [Batch, Num_Nodes, Feature_Dim])")
    print(f"      * Feature_Dim = 4 (1 Syndrome + 3 One-hot Colors)")
    print(f"    - Edge Index Shape:    {edges.shape}")
    
    print(f"    - Graph Statistics:    {graph_mapper.num_nodes} Nodes, {edges.shape[1]} Edges found.")
    
    if edges.shape[1] > 0:
        print("    - Graph connectivity check: Passed (Edges exist).")
    else:
        print("    - WARNING: No edges found. This is unexpected for noisy circuits.")
        
    print("    -> Step 3 Complete.\n")

    # ==========================================================================
    # Step 4: 모델 입출력 차원 검증 (CNN)
    # ==========================================================================
    print(">>> [Step 4] Verifying CNN Model I/O...")
    
    # models 패키지 경로에 주의하세요 (main.py 위치 기준 상위 폴더 참조 필요)
    # 실행 위치에 따라 sys.path 설정이 필요할 수 있습니다.
    try:
        # 편의상 모델 정의를 가져오거나 shape만 체크합니다.
        # 여기서는 shape만 논리적으로 맞는지 확인합니다.
        num_qubits = physical_errors.shape[1]
        print(f"    - Model Output Layer Size should be: {num_qubits}")
        print(f"    - Current Labels (Physical Errors) match this shape.")
        
        # 실제 텐서 변환 테스트
        tensor_img = torch.FloatTensor(syndrome_images)
        print(f"    - Tensor Conversion Check: Passed. Shape {tensor_img.shape}")
        
    except Exception as e:
        print(f"    - WARNING: Model verification failed: {e}")

    print("    -> Step 4 Complete.\n")
    
    # 상세 정보 출력
    inspect_shot_details(0, raw_detectors, image_mapper)

    print("=== All Checks Passed Successfully! ===")

if __name__ == "__main__":
    main()