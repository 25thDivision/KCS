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

from simulation.common.dataset import QECDataset
from models.cnn import CNN

# ==============================================================================
# 설정 (CNN 전용)
# ==============================================================================
MODEL_NAME = "CNN"
DISTANCE = 5
ERROR_RATE = 0.05
ERROR_TYPE = "Z"

DATASET_DIR = "dataset/color_code/image"
TRAIN_FILE = f"train_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"
TEST_FILE  = f"test_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"

MODEL_SAVE_DIR = "saved_weights/cnn"
CHECKPOINT_NAME = f"checkpoint_cnn_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.pth"
BEST_MODEL_NAME = f"best_cnn_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.pth"

# [수정] 결과를 test_results 폴더에 모델별로 따로 저장
RESULT_LOG_FILE = "test_results/benchmark_results_CNN.csv"

# [하이퍼파라미터]
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 3
OPTIMIZER_NAME = "Adam"

# [모델 구조 정보 (기록용)]
KERNEL_SIZE = "3x3"
LAYERS_INFO = "3 (1 Conv + 2 FC)"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}\n simulation/generate_dataset_image.py를 실행했는지 확인하세요.")
    data = np.load(path)
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
        print(f">>> 🔄 체크포인트 발견! 학습을 재개합니다: {path}")
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_ecr = checkpoint['best_ecr']
        patience_counter = checkpoint['patience_counter']
        
        print(f"    - 시작 에포크: {start_epoch}")
        print(f"    - 현재 최고 ECR: {best_ecr:.2%}")
        return start_epoch, best_ecr, patience_counter
    else:
        print(">>> 🆕 체크포인트가 없습니다. 처음부터 시작합니다.")
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

def log_results(model_name, d, p, err_type, ecr, acc, inf_time, 
                epochs, lr, batch, opt, kern, lay):
    
    # [안전장치] 저장할 폴더가 없으면 만듭니다.
    log_dir = os.path.dirname(RESULT_LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_exists = os.path.isfile(RESULT_LOG_FILE)
    
    with open(RESULT_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = [
                "Model", "Distance", "Error_Rate(p)", "Error_Type", 
                "Best_ECR(%)", "Accuracy(%)", "Inference_Time(ms)",
                "Max_Epochs", "Learning_Rate", "Batch_Size", "Optimizer", 
                "Kernel_Size", "Layers"
            ]
            writer.writerow(headers)
        
        writer.writerow([
            model_name, d, p, err_type, 
            f"{ecr:.2f}", f"{acc:.2f}", f"{inf_time:.4f}",
            epochs, lr, batch, opt, kern, lay
        ])
    
    print(f"\n>>> 📝 상세 결과가 '{RESULT_LOG_FILE}'에 저장되었습니다.")

def main():
    print(f"=== {MODEL_NAME} Training (d={DISTANCE}, p={ERROR_RATE}, Type={ERROR_TYPE}) ===")
    
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNN(
        height=X_train.shape[2], 
        width=X_train.shape[3], 
        in_channels=1, 
        num_classes=y_train.shape[1]
    ).to(DEVICE)
    
    pos_weight_val = (1.0 - ERROR_RATE) / ERROR_RATE
    pos_weight = torch.tensor([pos_weight_val] * y_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    start_epoch, best_ecr, patience_counter = load_checkpoint(model, optimizer, CHECKPOINT_NAME)
    
    best_inf_time = 0.0
    best_acc = 0.0

    print(f">>> 학습 시작 ({start_epoch+1} ~ {MAX_EPOCHS} Epochs)...")
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ecr, val_acc, val_time = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
              f"Loss: {train_loss:.4f} | ECR: {val_ecr:.2%} | Acc: {val_acc:.2%} | Patience: {patience_counter}/{PATIENCE}")
        
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            best_acc = val_acc
            best_inf_time = val_time
            patience_counter = 0
            
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME))
            save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, CHECKPOINT_NAME)
            print(f"    -> 👑 최고 기록 갱신! (ECR: {best_ecr:.2%})")
        else:
            patience_counter += 1
            save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, CHECKPOINT_NAME)
            
            if patience_counter >= PATIENCE:
                print(f"\n>>> 🛑 Early Stopping 발동! (Epoch {epoch+1})")
                break

    print(f"\n>>> 학습 종료. 최종 Best ECR: {best_ecr:.2%}")
    
    log_results(
        MODEL_NAME, DISTANCE, ERROR_RATE, ERROR_TYPE, 
        best_ecr*100, best_acc*100, best_inf_time,
        MAX_EPOCHS, LEARNING_RATE, BATCH_SIZE, OPTIMIZER_NAME, 
        KERNEL_SIZE, LAYERS_INFO
    )

if __name__ == "__main__":
    main()