import os
import sys
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed  # [추가] 병렬 처리 라이브러리

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generators.color_code import create_color_code_circuit, generate_dataset
from common.mapper_image import SyndromeImageMapper 

# ==============================================================================
# [설정] 대용량 데이터 생성 (Train + Test)
# ==============================================================================
# 1. 학습 데이터 (Train) 개수
TRAIN_SAMPLES = {
    3: 10000000, 
    5: 1000000, 
    7: 1000000
}

# 2. 테스트 데이터 (Test) 개수
TEST_SAMPLES = {
    3: 100000,
    5: 100000,
    7: 100000
}

# [수정] 한 번에 처리할 작업 단위 (너무 크면 메모리 터짐, 너무 작으면 관리 비용 듬)
# 48코어 서버라면 2000~5000 정도가 적당합니다.
CHUNK_SIZE = 5000  

# [수정] 병렬 처리에 사용할 CPU 코어 수 (-1: 모든 코어 사용)
NUM_WORKERS = -1

NOISE_RATES = [0.005, 0.01, 0.05]
ERROR_TYPES = ["X", "Z"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/color_code/image")

# [추가] 병렬 처리를 위한 단위 작업 함수 (이 함수가 각 CPU 코어로 복사되어 실행됨)
def _generate_chunk(d, p, err_type, count):
    """
    작은 단위(Chunk)의 데이터를 생성하여 반환합니다.
    """
    # 기존 generate_dataset 함수 사용
    raw, phys = generate_dataset(d, d, p, count, error_type=err_type)
    return raw, phys

def generate_and_save(mapper, d, p, err_type, total_samples, file_prefix):
    """데이터를 병렬로 생성하고 저장하는 함수"""
    
    # 1. 작업을 잘게 쪼갭니다 (총 개수 / 청크 크기)
    num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
    
    desc = f"       [{file_prefix.upper()}] ({err_type}, p={p})"
    print(f"{desc} -> Generating with {NUM_WORKERS} workers...")

    # 2. [핵심] 병렬 실행 (Parallel)
    # joblib이 알아서 CHUNK_SIZE만큼씩 나눠서 48개 코어에 뿌려줍니다.
    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(_generate_chunk)(
            d, p, err_type, 
            min(CHUNK_SIZE, total_samples - i * CHUNK_SIZE)
        )
        for i in tqdm(range(num_chunks), desc="       Processing Chunks")
    )
    
    # 3. 결과 합치기 및 이미지 매핑
    # 생성은 병렬로 했지만, 저장은 한 번에 해야 하므로 모아서 처리합니다.
    print("       -> Mapping to Images & Merging...")
    
    all_images = []
    all_labels = []

    # 병렬 처리된 결과물들을 하나씩 꺼내서 이미지로 변환
    for raw_detectors, physical_errors in results:
        # 매핑 자체는 금방 끝나므로 메인 프로세스에서 수행
        images = mapper.map_to_images(raw_detectors)
        all_images.append(images)
        all_labels.append(physical_errors)
    
    # 리스트를 하나의 거대한 Numpy 배열로 병합
    full_images = np.concatenate(all_images, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    
    # 저장
    file_name = f"{file_prefix}_d{d}_p{p}_{err_type}.npz"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    np.savez_compressed(
        file_path,
        features=full_images,        
        labels=full_labels  
    )
    print(f"       Saved: {file_name} (Shape: {full_images.shape})")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"=== Generating Image Datasets (Parallel Mode: {NUM_WORKERS}) ===")
    
    for d, train_count in TRAIN_SAMPLES.items():
        test_count = TEST_SAMPLES[d]
        print(f"\n>>> Processing Distance d={d} (Train: {train_count}, Test: {test_count})")
        
        circuit = create_color_code_circuit(d, d, 0.001)
        mapper = SyndromeImageMapper(circuit)
        
        for p in NOISE_RATES:
            for err_type in ERROR_TYPES:
                print(f"    -> Processing {err_type}-Error (p={p})...")
                
                # 1. Train 데이터 생성
                generate_and_save(mapper, d, p, err_type, train_count, "train")
                
                # 2. Test 데이터 생성
                generate_and_save(mapper, d, p, err_type, test_count, "test")

    print("\n=== All Image Datasets Generated Successfully! ===")

if __name__ == "__main__":
    main()