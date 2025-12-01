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
DATASET_DIR = "dataset/color_code/image"  # 이미지 데이터 경로 고정!
TRAIN_FILE = "train_d5_p0.05.npz"
TEST_FILE  = "test_d5_p0.05.npz"

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
            
            preds = (outputs > 0).float()
            error_mask = (labels == 1)
            total_error_bits += error_mask.sum().item()
            detected_error_bits += (preds[error_mask] == 1).sum().item()
            
    avg_loss = total_loss / len(loader)
    ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
    return avg_loss, ecr

def main():
    print(f"=== CNN Training (d=5, p=0.05) ===")
    
    # 1. 데이터 로드
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    print(f"    Train Shape: {X_train.shape}") # (N, 1, 6, 5) 예상
    
    # 2. 로더 생성
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 모델 초기화 (CNN)
    height = X_train.shape[2]
    width = X_train.shape[3]
    num_qubits = y_train.shape[1]
    
    model = CNN(height, width, in_channels=1, num_classes=num_qubits).to(DEVICE)
    
    # 4. 학습
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(">>> Starting Training...")
    best_ecr = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ecr = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ECR: {val_ecr:.2%}")
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            # torch.save(model.state_dict(), "best_cnn.pth")

    print(f"\n>>> CNN Training Complete. Best ECR: {best_ecr:.2%}")

if __name__ == "__main__":
    main()