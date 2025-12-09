import os
import sys
import numpy as np
from tqdm import tqdm

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

# 2. 테스트 데이터 (Test) 개수 (검증용)
# 보통 10만 개 정도면 충분히 정확한 통계를 얻을 수 있습니다.
TEST_SAMPLES = {
    3: 100000,
    5: 100000,
    7: 100000
}

BATCH_SIZE = 100000  # 배치 크기

NOISE_RATES = [0.005, 0.01, 0.05]
ERROR_TYPES = ["X", "Z"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/color_code/image")

def generate_and_save(mapper, d, p, err_type, total_samples, file_prefix):
    """데이터를 생성하고 저장하는 공통 함수"""
    all_images = []
    all_labels = []
    
    num_batches = int(np.ceil(total_samples / BATCH_SIZE))
    
    # 진행바 설명 (Train인지 Test인지 표시)
    desc = f"       [{file_prefix.upper()}] ({err_type}, p={p})"
    
    for b in tqdm(range(num_batches), desc=desc):
        current_batch_size = min(BATCH_SIZE, total_samples - b * BATCH_SIZE)
        
        # 데이터 생성
        raw_detectors, physical_errors = generate_dataset(
            d, d, p, current_batch_size, error_type=err_type
        )
        
        # 이미지 변환
        images = mapper.map_to_images(raw_detectors)
        
        all_images.append(images)
        all_labels.append(physical_errors)
    
    # 병합
    full_images = np.concatenate(all_images, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    
    # 저장 (train_... 또는 test_...)
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
        
    print(f"=== Generating Image Datasets (Train & Test) ===")
    
    for d, train_count in TRAIN_SAMPLES.items():
        test_count = TEST_SAMPLES[d]
        print(f"\n>>> Processing Distance d={d} (Train: {train_count}, Test: {test_count})")
        
        circuit = create_color_code_circuit(d, d, 0.001)
        mapper = SyndromeImageMapper(circuit)
        
        for p in NOISE_RATES:
            for err_type in ERROR_TYPES:
                print(f"    -> Processing {err_type}-Error (p={p})...")
                
                # 1. Train 데이터 생성
                # (만약 Train이 이미 있고 Test만 만들고 싶다면 이 줄을 주석 처리하세요)
                # generate_and_save(mapper, d, p, err_type, train_count, "train")
                
                # 2. Test 데이터 생성 (필수!)
                generate_and_save(mapper, d, p, err_type, test_count, "test")

    print("\n=== All Image Datasets Generated Successfully! ===")

if __name__ == "__main__":
    main()