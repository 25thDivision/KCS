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
# GNN 모델
from models.gnn import GNN

# ==============================================================================
# 설정 (GNN 전용)
# ==============================================================================
MODEL_NAME = "GNN"
DISTANCE = 5
ERROR_RATE = 0.05
DEPTH = 4  # [설정] GNN 레이어 깊이 (Depth)

DATASET_DIR = "dataset/color_code/graph"
TRAIN_FILE = f"train_d{DISTANCE}_p{ERROR_RATE}.npz"
TEST_FILE  = f"test_d{DISTANCE}_p{ERROR_RATE}.npz"
EDGE_FILE  = f"edges_d{DISTANCE}.npy"

MODEL_SAVE_DIR = "saved_weights/gnn"
MODEL_SAVE_NAME = f"gnn_d{DISTANCE}_p{ERROR_RATE}_depth{DEPTH}_best.pth"
RESULT_LOG_FILE = "benchmark_results.csv"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}\n simulation/generate_dataset_graph.py를 실행하세요.")
    data = np.load(path)
    return data['features'], data['labels']

def load_edges(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"엣지 파일 없음: {path}")
    edges = np.load(path)
    return torch.LongTensor(edges).to(DEVICE)

def train_one_epoch(model, loader, optimizer, criterion, device, edge_index):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training Epoch", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, edge_index) # 엣지 전달
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, edge_index):
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
            outputs = model(inputs, edge_index)
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            total_samples += inputs.size(0)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0).float()
            
            # ECR
            error_mask = (labels == 1)
            total_error_bits += error_mask.sum().item()
            detected_error_bits += (preds[error_mask] == 1).sum().item()
            
            # Accuracy
            correct_bits += (preds == labels).sum().item()
            total_bits += labels.numel()
            
    avg_loss = total_loss / len(loader)
    ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
    accuracy = correct_bits / total_bits if total_bits > 0 else 0.0
    avg_inference_time_ms = (total_inference_time / total_samples) * 1000
    
    return avg_loss, ecr, accuracy, avg_inference_time_ms

def log_results(model_name, d, p, depth, ecr, acc, inf_time):
    file_exists = os.path.isfile(RESULT_LOG_FILE)
    with open(RESULT_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 헤더에 Depth 추가
        if not file_exists:
            writer.writerow(["Model", "Distance", "Error_Rate(p)", "Depth", "Best_ECR(%)", "Accuracy(%)", "Inference_Time(ms)"])
        
        writer.writerow([model_name, d, p, depth, f"{ecr:.2f}", f"{acc:.2f}", f"{inf_time:.4f}"])
    
    print(f"\n>>> 📝 Result logged to '{RESULT_LOG_FILE}'")

def main():
    print(f"=== GNN Training (d={DISTANCE}, p={ERROR_RATE}, Depth={DEPTH}) on {DEVICE} ===")
    
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    edge_index = load_edges(EDGE_FILE)
    
    print(f"    Train: {X_train.shape}")
    print(f"    Edges: {edge_index.shape}")
    
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화 (Depth 적용)
    model = GNN(
        num_nodes=X_train.shape[1], 
        in_channels=X_train.shape[2], 
        num_qubits=y_train.shape[1],
        hidden_dim=64,
        num_layers=DEPTH  # <--- Depth 전달
    ).to(DEVICE)
    
    # 가중치 적용
    pos_weight_val = (1.0 - ERROR_RATE) / ERROR_RATE
    pos_weight = torch.tensor([pos_weight_val] * y_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    print(">>> Starting Training...")
    best_ecr = 0.0
    best_acc = 0.0
    best_inf_time = 0.0
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, edge_index)
        val_loss, val_ecr, val_acc, val_time = evaluate(model, test_loader, criterion, DEVICE, edge_index)
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
              f"Loss: {train_loss:.4f} | ECR: {val_ecr:.2%} | Acc: {val_acc:.2%} | Time: {val_time:.3f}ms")
        
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            best_acc = val_acc
            best_inf_time = val_time
            patience_counter = 0
            
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME))
            print(f"    -> 💾 New Best Saved! (ECR: {best_ecr:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n>>> 🛑 Early Stopping triggered.")
                break

    print(f"\n>>> Training Complete. Best ECR: {best_ecr:.2%}")
    # 결과 기록 (Depth 포함)
    log_results(MODEL_NAME, DISTANCE, ERROR_RATE, DEPTH, best_ecr*100, best_acc*100, best_inf_time)

if __name__ == "__main__":
    main()