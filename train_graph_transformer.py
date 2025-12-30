import os
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
# Graph Transformer 모델
from models.graph_transformer import GraphTransformer

# ==============================================================================
# 설정 (Graph Transformer)
# ==============================================================================
MODEL_NAME = "GraphTransformer"
DISTANCE = 3
ERROR_RATE = 0.005
ERROR_TYPE = "X"

NUM_WORKERS = 8

DATASET_DIR = "dataset/color_code/graph"
TRAIN_FILE = f"train_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"
TEST_FILE  = f"test_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"
EDGE_FILE = f"edges_d{DISTANCE}.npy"

MODEL_SAVE_DIR = "saved_weights/graph_transformer"
CHECKPOINT_NAME = f"checkpoint_gt_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.pth"
BEST_MODEL_NAME = f"best_gt_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.pth"
RESULT_LOG_FILE = "test_results/benchmark_results_GT.csv"

# [학습 하이퍼파라미터]
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
PATIENCE = 3
OPTIMIZER_NAME = "Adam"

# [모델 구조 하이퍼파라미터]
GT_D_MODEL = 256
GT_NUM_HEADS = 4
GT_NUM_LAYERS = 5
GT_DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    data = np.load(path)
    return data['features'], data['labels']

def save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, filename):
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
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
        print(f">>> 체크포인트 로드: {path}")
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'] + 1, checkpoint['best_ecr'], checkpoint['patience_counter']
        except Exception as e:
            print(f">>> 로드 실패: {e}")
            return 0, 0.0, 0
    else:
        print(">>> 체크포인트 없음. 처음부터 시작합니다.")
        return 0, 0.0, 0

def load_edges(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"엣지 파일 없음: {path}\n generate_dataset_graph.py를 먼저 실행하세요.")
    edges = np.load(path)
    # PyTorch LongTensor로 변환 및 GPU로 이동 준비
    return torch.LongTensor(edges).to(DEVICE)

def train_one_epoch(model, loader, optimizer, criterion, device, edge_index): 
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training Epoch", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # edge_index 전달
        outputs = model(inputs, edge_index) 
        
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

# [수정] 매 Epoch마다 로그를 저장하는 함수
def log_epoch_result(epoch, train_loss, val_loss, val_ecr, val_acc, val_time, lr):
    log_dir = os.path.dirname(RESULT_LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_exists = os.path.isfile(RESULT_LOG_FILE)
    
    # 파일을 append 모드로 엽니다.
    with open(RESULT_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 파일이 처음 생성될 때만 헤더 작성
        if not file_exists:
            headers = [
                "Distance", "Error_Rate(p)", "Error_Type", 
                "Best_ECR(%)", "Accuracy(%)", "Inference_Time(ms)", # Best_ECR(%) 컬럼에 현재 ECR 저장
                "Epochs", "Learning_Rate", "Train_Loss", "Val_Loss"
            ]
            writer.writerow(headers)
        
        # 데이터 작성
        writer.writerow([
            DISTANCE, ERROR_RATE, ERROR_TYPE,
            f"{val_ecr*100:.2f}",  # 현재 Epoch의 ECR (%)
            f"{val_acc*100:.2f}",  # 현재 Epoch의 Accuracy (%)
            f"{val_time:.4f}",     # Inference Time (ms)
            epoch,                 # 현재 Epoch
            lr,                    # Learning Rate
            f"{train_loss:.4f}"    # Train Loss
            f"{val_loss:.4f}"      # Val Loss 추가
        ])

def main():
    print(f"=== {MODEL_NAME} Training (d={DISTANCE}, p={ERROR_RATE}, Type={ERROR_TYPE}) ===")
    
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    edge_index = load_edges(EDGE_FILE)
    
    train_loader = DataLoader(
        QECDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        QECDataset(X_test, y_test), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    model = GraphTransformer(
        num_nodes=X_train.shape[1], 
        in_channels=X_train.shape[2], 
        num_qubits=y_train.shape[1],
        d_model=GT_D_MODEL,
        num_heads=GT_NUM_HEADS,
        num_layers=GT_NUM_LAYERS,
        dropout=GT_DROPOUT
    ).to(DEVICE)
    
    # pos_weight 계산 (클래스 불균형 처리)
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
        # 1. 학습
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, edge_index)
        
        # 2. 평가
        val_loss, val_ecr, val_acc, val_time = evaluate(model, test_loader, criterion, DEVICE, edge_index)
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
              f"Loss(T/V): {train_loss:.4f}/{val_loss:.4f} | "  # Train/Val Loss 비교
              f"ECR: {val_ecr:.2%} | Acc: {val_acc:.2%} | "
              f"Time: {val_time:.2f}ms | "                       # 추론 시간 표시
              f"Patience: {patience_counter}/{PATIENCE}")
        
        # 3. [추가] 로그 저장 (매 Epoch 마다)
        log_epoch_result(epoch + 1, train_loss, val_loss, val_ecr, val_acc, val_time, LEARNING_RATE)

        # 4. 체크포인트 및 Early Stopping
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

    print(f"\n>>> 학습 종료. Best ECR: {best_ecr:.2%}, Acc: {best_acc:.2%}, Time: {best_inf_time:.4f}ms")
    # 마지막 요약 로그는 더 이상 중복 저장하지 않음 (루프 안에서 이미 다 저장됨)

if __name__ == "__main__":
    main()