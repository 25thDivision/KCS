import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# [Import] 공통 데이터셋 모듈
# (simulation 폴더가 KCS 폴더 안에 있어야 합니다)
from simulation.common.dataset import QECDataset

# [Import] Graph Transformer 모델
# 아직 models/graph_transformer.py가 없다면 에러가 날 수 있으니,
# 파일을 만든 후에 아래 주석을 해제하세요.
# from models.graph_transformer import GraphTransformer

# ==============================================================================
# 설정 (Graph 전용)
# ==============================================================================
# 그래프 데이터가 저장된 경로 (generate_dataset.py의 저장 위치와 일치해야 함)
DATASET_DIR = "dataset/color_code/graph"

# 파일명
TRAIN_FILE = "train_d5_p0.05.npz"
TEST_FILE  = "test_d5_p0.05.npz"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    """ .npz 파일에서 feature와 label을 로드합니다 """
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}\n 먼저 simulation/generate_dataset.py를 실행하여 그래프 데이터를 생성해주세요.")
        
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
    
    # ECR (Error Correction Rate) 계산 변수
    total_error_bits = 0
    detected_error_bits = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 예측: Logit > 0 이면 에러(1)로 판단
            preds = (outputs > 0).float()
            
            # 실제 에러 위치(1) 확인
            error_mask = (labels == 1)
            total_error_bits += error_mask.sum().item()
            
            # Recall (실제 에러 중 맞춘 개수)
            detected_error_bits += (preds[error_mask] == 1).sum().item()
            
    avg_loss = total_loss / len(loader)
    # 분모가 0일 경우 방지
    ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
    
    return avg_loss, ecr

def main():
    print(f"=== Graph Transformer Training (Device: {DEVICE}) ===")
    
    # 1. 데이터 로드
    print(f">>> Loading Graph Data from '{DATASET_DIR}'...")
    try:
        X_train, y_train = load_data(TRAIN_FILE)
        X_test, y_test = load_data(TEST_FILE)
        # Graph Data Shape: [N, Nodes, Features]
        print(f"    Train Shape: {X_train.shape}")
        print(f"    Test Shape:  {X_test.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 2. 데이터셋 및 로더 생성
    train_dataset = QECDataset(X_train, y_train)
    test_dataset = QECDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 모델 초기화
    print(">>> Initializing Graph Transformer Model...")
    
    # 입력 차원 정보 추출
    num_nodes = X_train.shape[1]   # 예: 45
    in_channels = X_train.shape[2] # 예: 6
    num_qubits = y_train.shape[1]  # 예: 28
    
    # [주의] models/graph_transformer.py 파일을 만든 후 아래 주석을 풀어야 합니다.
    # model = GraphTransformer(
    #     num_nodes=num_nodes,
    #     in_channels=in_channels,
    #     num_qubits=num_qubits
    # ).to(DEVICE)
    
    print("⚠️ 아직 GraphTransformer 모델 파일이 없습니다. models/graph_transformer.py를 구현하고 주석을 해제하세요.")
    return # 모델이 없으므로 여기서 종료합니다. (구현 후 삭제하세요)

    # 4. 학습 루프 (모델 연결 후 실행됨)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(">>> Starting Training...")
    best_ecr = 0.0
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ecr = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ECR (Recall): {val_ecr:.2%}")
        
        if val_ecr > best_ecr:
            best_ecr = val_ecr
            # 모델 저장 (선택)
            # torch.save(model.state_dict(), "best_graph_model.pth")

    print(f"\n>>> Training Complete. Best ECR: {best_ecr:.2%}")

if __name__ == "__main__":
    main()