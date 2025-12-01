import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# [Import] 공통 데이터셋 모듈
from simulation.common.dataset import QECDataset
# [Import] CNN 모델
from models.cnn import CNN

# ==============================================================================
# 설정 (CNN 전용)
# ==============================================================================
DATASET_DIR = "dataset/color_code/image"  # 이미지 데이터 경로
TRAIN_FILE = "train_d5_p0.05.npz"
TEST_FILE  = "test_d5_p0.05.npz"

# 저장할 모델 폴더 및 파일명 설정
MODEL_SAVE_DIR = "saved_weights/cnn"
MODEL_SAVE_NAME = "cnn_d5_p0.05_best.pth"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}\n simulation/generate_data_image.py를 먼저 실행하세요.")
    data = np.load(path)
    return data['features'], data['labels']

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for inputs, labels in tqdm(loader, desc="Training Epoch", leave=False):
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
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 예측: Logit이 0보다 크면 에러(1)로 판단
            preds = (outputs > 0).float()
            
            # 실제 에러가 있는 위치(1)만 골라냅니다
            error_mask = (labels == 1)
            total_error_bits += error_mask.sum().item()
            
            # 실제 에러가 있는 곳 중에서, 예측도 1인 개수 (Recall)
            detected_error_bits += (preds[error_mask] == 1).sum().item()
            
    avg_loss = total_loss / len(loader)
    # 분모가 0일 경우(에러가 하나도 없는 경우) 방지
    ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
    
    return avg_loss, ecr

def main():
    print(f"=== CNN Training (d=5, p=0.05) on {DEVICE} ===")
    
    # 1. 데이터 로드
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    print(f"    Train Shape: {X_train.shape}")
    print(f"    Test Shape:  {X_test.shape}")
    
    # 2. 로더 생성
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 모델 초기화 (CNN)
    height = X_train.shape[2]
    width = X_train.shape[3]
    num_qubits = y_train.shape[1]
    
    model = CNN(height, width, in_channels=1, num_classes=num_qubits).to(DEVICE)
    
    # 기존 학습된 가중치 불러오기
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
    
    if os.path.exists(save_path):
        print(f">>> 🔄 Found existing model: {save_path}")
        print(">>> Loading weights and resuming training...")
        try:
            model.load_state_dict(torch.load(save_path))
            print(">>> ✅ Weights loaded successfully!")
        except Exception as e:
            print(f">>> ⚠️ Failed to load weights: {e}")
            print(">>> Starting from scratch.")
    else:
        print(">>> No existing model found. Starting from scratch.")
    
    # [설정] 클래스 불균형 해결을 위한 pos_weight (선택 사항, 필요시 활성화)
    # p=0.05 기준, 0이 1보다 약 19배 많음 -> 19.0배 가중치 부여
    pos_weight_val = (1.0 - 0.05) / 0.05 
    pos_weight = torch.tensor([pos_weight_val] * num_qubits).to(DEVICE)
    
    # 가중치 적용된 손실 함수
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # [추가] 모델 저장 폴더 생성
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f">>> Created directory: {MODEL_SAVE_DIR}")

    # 4. 학습 루프
    print(">>> Starting Training...")
    best_ecr = 0.0
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ecr = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ECR (Recall): {val_ecr:.2%}")
        
        # [추가] Best Model 저장 로직
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            save_path = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"    -> 💾 New Best Model Saved! (ECR: {best_ecr:.2%}) at {save_path}")

    print(f"\n>>> CNN Training Complete. Best ECR: {best_ecr:.2%}")

if __name__ == "__main__":
    main()