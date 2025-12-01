import os
import sys
import numpy as np
from tqdm import tqdm

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Image Mapper Import
from generators.color_code import create_color_code_circuit, generate_dataset
from common.mapper_image import SyndromeImageMapper 

# 설정
DISTANCE = 5
ROUNDS = 5
NOISE_RATE_TRAIN = 0.05
NUM_TRAIN = 100000
NUM_TEST = 10000

# [핵심] 이미지 데이터 전용 폴더 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/color_code/image")

def save_dataset(mode, num_samples, noise_rate, mapper):
    print(f"\n>>> Generating {mode} set (Image) ({num_samples} samples, p={noise_rate})...")
    
    # 1. 데이터 생성
    raw_detectors, physical_errors = generate_dataset(DISTANCE, ROUNDS, noise_rate, num_samples)
    
    # 2. 이미지 변환
    print(f"    - Converting to Image Features...")
    images = mapper.map_to_images(raw_detectors)
    
    # 3. 저장 (파일명에 _image 제거! 깔끔하게!)
    file_name = f"{mode}_d{DISTANCE}_p{noise_rate}.npz"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    np.savez_compressed(
        file_path,
        features=images,        
        labels=physical_errors  
    )
    print(f"    - Saved to {file_path}")
    print(f"      X shape: {images.shape}, Y shape: {physical_errors.shape}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"=== Generating Image Dataset for CNN (d={DISTANCE}) ===")
    print(">>> Initializing Circuit and Mapper (Image Mode)...")
    circuit = create_color_code_circuit(DISTANCE, ROUNDS, 0.001)
    mapper = SyndromeImageMapper(circuit)
    
    save_dataset("train", NUM_TRAIN, NOISE_RATE_TRAIN, mapper)
    save_dataset("test", NUM_TEST, NOISE_RATE_TRAIN, mapper)
    
    print("\n=== Image Datasets Generated Successfully! ===")

if __name__ == "__main__":
    main()