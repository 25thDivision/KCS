import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv

# 공통 모듈
from simulation.common.dataset import QECDataset
# U-Net 모델
from models.unet import UNet

# ==============================================================================
# 설정 (U-Net 전용)
# ==============================================================================
MODEL_NAME = "UNet"
DISTANCE = 3   # [주의] 생성된 이미지 데이터셋의 Distance와 일치해야 함
ERROR_RATE = 0.05
ERROR_TYPE = "X"

# [데이터셋 경로: CNN과 동일한 Image 데이터셋 사용]
DATASET_DIR = "dataset/color_code/image"
TRAIN_FILE = f"train_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"
TEST_FILE  = f"test_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"

# [저장 경로]
MODEL_SAVE_DIR = "saved_weights/unet"
CHECKPOINT_NAME = f"checkpoint_unet_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.pth"
BEST_MODEL_NAME = f"best_unet_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.pth"
RESULT_LOG_FILE = "test_results/benchmark_results_UNet.csv"

# [학습 하이퍼파라미터]
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 3
OPTIMIZER_NAME = "Adam"

# ALPHA = (1.0 - ERROR_RATE) / ERROR_RATE
ALPHA = 8.0
BETA = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}\n simulation/generate_dataset_image.py를 실행했는지 확인하세요.")
    data = np.load(path)
    # 이미지 데이터는 (N, 1, H, W) 형태여야 함
    return data['features'], data['labels']

def save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_ecr': best_ecr,
        'patience_counter': patience_counter
    }
    path = os.path.join(MODEL_SAVE_DIR, filename)
    torch.save(state, path)

def load_checkpoint(model, optimizer, filename):
    path = os.path.join(MODEL_SAVE_DIR, filename)
    if os.path.exists(path):
        print(f">>> 🔄 체크포인트 발견! {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['best_ecr'], checkpoint['patience_counter']
    else:
        print(">>> 🆕 체크포인트 없음. 새로 시작합니다.")
        return 0, 0.0, 0

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
    
    total_inference_time = 0.0
    total_samples = 0
    correct_bits = 0
    total_bits = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            total_samples += inputs.size(0)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0).float()
            
            error_mask = (labels == 1)
            total_error_bits += error_mask.sum().item()
            detected_error_bits += (preds[error_mask] == 1).sum().item()
            
            correct_bits += (preds == labels).sum().item()
            total_bits += labels.numel()
            
    avg_loss = total_loss / len(loader)
    ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
    accuracy = correct_bits / total_bits if total_bits > 0 else 0.0
    avg_inference_time_ms = (total_inference_time / total_samples) * 1000
    
    return avg_loss, ecr, accuracy, avg_inference_time_ms

def log_results(model_name, d, p, err_type, ecr, acc, inf_time, epochs, lr):
    log_dir = os.path.dirname(RESULT_LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_exists = os.path.isfile(RESULT_LOG_FILE)
    with open(RESULT_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Distance", "Error_Rate(p)", "Error_Type", "Best_ECR(%)", "Accuracy(%)", "Inference_Time(ms)", "Epochs", "LR"])
        writer.writerow([model_name, d, p, err_type, f"{ecr:.2f}", f"{acc:.2f}", f"{inf_time:.4f}", epochs, lr])
    print(f"\n>>> 📝 결과 저장 완료: {RESULT_LOG_FILE}")

def main():
    print(f"=== {MODEL_NAME} Training (d={DISTANCE}, p={ERROR_RATE}) ===")
    
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    # [🚨 긴급 추가] 데이터가 너무 많으면 잘라서 씁니다 (시간 단축용)
    # 1000만 개 다 쓰지 말고 5만 개만 써도 d=3에서는 충분히 수렴합니다.
    if X_train.shape[0] > 50000:
        print(f">>> ✂️ 데이터가 너무 많아 50,000개로 자릅니다. (원본: {X_train.shape[0]})")
        X_train = X_train[:50000]
        y_train = y_train[:50000]
    
    # U-Net 입력 차원 확인 (B, 1, H, W)
    H, W = X_train.shape[2], X_train.shape[3]
    num_qubits = y_train.shape[1]
    
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNet(height=H, width=W, in_channels=1, num_classes=num_qubits).to(DEVICE)
    
    pos_weight = torch.tensor([ALPHA] * num_qubits).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    start_epoch, best_ecr, patience_counter = load_checkpoint(model, optimizer, CHECKPOINT_NAME)
    best_inf_time = 0.0
    best_acc = 0.0

    print(f">>> 학습 시작 ({start_epoch+1} ~ {MAX_EPOCHS})...")
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ecr, val_acc, val_time = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Loss: {train_loss:.4f} | ECR: {val_ecr:.2%} | Acc: {val_acc:.2%}")
        
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            best_acc = val_acc
            best_inf_time = val_time
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME))
            save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, CHECKPOINT_NAME)
            print(f"    -> 👑 Best ECR Updated: {best_ecr:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(">>> Early Stopping")
                break
                
    log_results(MODEL_NAME, DISTANCE, ERROR_RATE, ERROR_TYPE, best_ecr*100, best_acc*100, best_inf_time, MAX_EPOCHS, LEARNING_RATE)

if __name__ == "__main__":
    main()