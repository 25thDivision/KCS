import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv  # 결과 저장을 위해 추가

# [Import] 공통 데이터셋 모듈
from simulation.common.dataset import QECDataset
# [Import] CNN 모델
from models.cnn import CNN

# ==============================================================================
# 설정 (CNN 전용)
# ==============================================================================
MODEL_NAME = "CNN"  # 기록용 이름
DISTANCE = 5
ERROR_RATE = 0.05   # p값 (기록용)

DATASET_DIR = "dataset/color_code/image"
TRAIN_FILE = f"train_d{DISTANCE}_p{ERROR_RATE}.npz"
TEST_FILE  = f"test_d{DISTANCE}_p{ERROR_RATE}.npz"

MODEL_SAVE_DIR = "saved_weights/cnn"
MODEL_SAVE_NAME = f"cnn_d{DISTANCE}_p{ERROR_RATE}_best.pth"
RESULT_LOG_FILE = "benchmark_results.csv"  # 결과가 쌓일 파일

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100        # 최대 에포크 (넉넉하게)
PATIENCE = 10           # 10번 동안 성능 향상 없으면 조기 종료 (Early Stopping)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    data = np.load(path)
    return data['features'], data['labels']

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_error_bits = 0
    detected_error_bits = 0
    
    # 추론 시간 측정 변수
    total_inference_time = 0.0
    total_samples = 0
    
    # Accuracy 측정을 위한 변수 (전체 큐비트 중 맞춘 비율)
    correct_bits = 0
    total_bits = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # [시간 측정 시작]
            start_time = time.time()
            outputs = model(inputs)
            # [시간 측정 끝]
            end_time = time.time()
            
            # 배치 전체 추론 시간 누적
            total_inference_time += (end_time - start_time)
            total_samples += inputs.size(0)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 예측 변환
            preds = (outputs > 0).float()
            
            # 1. ECR 계산 (Recall)
            error_mask = (labels == 1)
            total_error_bits += error_mask.sum().item()
            detected_error_bits += (preds[error_mask] == 1).sum().item()
            
            # 2. Accuracy 계산
            correct_bits += (preds == labels).sum().item()
            total_bits += labels.numel()
            
    avg_loss = total_loss / len(loader)
    ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
    accuracy = correct_bits / total_bits if total_bits > 0 else 0.0
    
    # 평균 추론 시간 (ms per sample) - 배치가 아닌 샘플 1개당 시간
    avg_inference_time_ms = (total_inference_time / total_samples) * 1000
    
    return avg_loss, ecr, accuracy, avg_inference_time_ms

def log_results(model_name, d, p, ecr, acc, inf_time):
    """결과를 CSV 파일에 한 줄 추가합니다."""
    file_exists = os.path.isfile(RESULT_LOG_FILE)
    
    with open(RESULT_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 파일이 처음 생기면 헤더 작성
        if not file_exists:
            writer.writerow(["Model", "Distance", "Error_Rate(p)", "Best_ECR(%)", "Accuracy(%)", "Inference_Time(ms)"])
        
        writer.writerow([model_name, d, p, f"{ecr:.2f}", f"{acc:.2f}", f"{inf_time:.4f}"])
    
    print(f"\n>>> 📝 Result logged to '{RESULT_LOG_FILE}'")

def main():
    print(f"=== {MODEL_NAME} Training (d={DISTANCE}, p={ERROR_RATE}) on {DEVICE} ===")
    
    # 1. 데이터 로드
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 모델 초기화
    model = CNN(
        height=X_train.shape[2], 
        width=X_train.shape[3], 
        in_channels=1, 
        num_classes=y_train.shape[1]
    ).to(DEVICE)
    
    # 가중치 적용
    pos_weight_val = (1.0 - ERROR_RATE) / ERROR_RATE
    pos_weight = torch.tensor([pos_weight_val] * y_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # 3. 학습 루프 (Early Stopping 적용)
    print(f">>> Starting Training (Max Epochs: {MAX_EPOCHS}, Patience: {PATIENCE})...")
    
    best_ecr = 0.0
    best_acc = 0.0
    best_inf_time = 0.0
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ecr, val_acc, val_time = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
              f"Loss: {train_loss:.4f} | ECR: {val_ecr:.2%} | Acc: {val_acc:.2%} | Time: {val_time:.3f}ms")
        
        # Best Model 갱신
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            best_acc = val_acc
            best_inf_time = val_time
            patience_counter = 0  # 카운터 초기화
            
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME))
            print(f"    -> 💾 New Best Saved! (ECR: {best_ecr:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n>>> 🛑 Early Stopping triggered after {epoch+1} epochs.")
                break

    print(f"\n>>> Training Finished. Best ECR: {best_ecr:.2%}")
    
    # 4. 결과 기록
    log_results(MODEL_NAME, DISTANCE, ERROR_RATE, best_ecr*100, best_acc*100, best_inf_time)

if __name__ == "__main__":
    main()