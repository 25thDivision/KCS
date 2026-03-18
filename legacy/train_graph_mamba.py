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
# 🆕 Graph Mamba 모델 임포트
from models.graph_mamba import GraphMamba

# ==============================================================================
# 설정 (Graph Mamba)
# ==============================================================================
MODEL_NAME = "GraphMamba"
DATASET_DIR = "dataset/color_code/graph"
MODEL_SAVE_DIR = "saved_weights/graph_mamba"
RESULT_LOG_FILE = "test_results/benchmark_results_Mamba.csv"

# [자동 학습을 위한 리스트 설정]
DISTANCES = [3, 5, 7]
ERROR_RATES = [0.005, 0.01, 0.05]
# Z 에러 데이터셋 문제가 해결되지 않았다면 ["X"]만 사용하는 것을 권장합니다.
ERROR_TYPES = ["X", "Z"] 

NUM_WORKERS = 8

# [학습 하이퍼파라미터]
# Mamba는 메모리 효율이 좋으므로 배치를 Transformer보다 더 키울 수 있습니다.
BATCH_SIZE = 1024 
LEARNING_RATE = 1e-4  
MAX_EPOCHS = 20
PATIENCE = 3
OPTIMIZER_NAME = "Adam"

# [Mamba 모델 구조 하이퍼파라미터]
GM_D_MODEL = 64      # 논문이나 보통 Mamba에서는 64~128 정도도 충분히 강력함
GM_NUM_LAYERS = 4    # Mamba 블록 깊이
GM_DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    data = np.load(path)
    return data['features'], data['labels']

def load_edges(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"엣지 파일 없음: {path}\n generate_dataset_graph.py를 먼저 실행하세요.")
    edges = np.load(path)
    return torch.LongTensor(edges).to(DEVICE)

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

def train_one_epoch(model, loader, optimizer, criterion, device, edge_index): 
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Graph Mamba는 forward에서 edge_index를 받아 정렬(Sorting)에 사용합니다.
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

def log_epoch_result(distance, error_rate, error_type, epoch, train_loss, val_loss, val_ecr, val_acc, val_time, lr):
    log_dir = os.path.dirname(RESULT_LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_exists = os.path.isfile(RESULT_LOG_FILE)
    
    with open(RESULT_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = [
                "Distance", "Error_Rate(p)", "Error_Type", 
                "Best_ECR(%)", "Accuracy(%)", "Inference_Time(ms)",
                "Epochs", "Learning_Rate", "Train_Loss", "Val_Loss"
            ]
            writer.writerow(headers)
        
        writer.writerow([
            distance, error_rate, error_type,
            f"{val_ecr*100:.2f}", 
            f"{val_acc*100:.2f}", 
            f"{val_time:.4f}",    
            epoch,                 
            lr,                    
            f"{train_loss:.4f}",   
            f"{val_loss:.4f}"      
        ])

def main():
    # 3중 Loop로 모든 설정 순회
    for distance in DISTANCES:
        for error_rate in ERROR_RATES:
            for error_type in ERROR_TYPES:
                
                print(f"\n{'='*60}")
                print(f"=== [GraphMamba] Training: d={distance}, p={error_rate}, Type={error_type} ===")
                print(f"{'='*60}")
                
                # 파일 경로 생성
                train_file = f"train_d{distance}_p{error_rate}_{error_type}.npz"
                test_file = f"test_d{distance}_p{error_rate}_{error_type}.npz"
                edge_file = f"edges_d{distance}.npy"
                
                # 파일 존재 여부 확인
                if not os.path.exists(os.path.join(DATASET_DIR, train_file)):
                    print(f">>> ⚠️ 데이터셋 없음: {train_file}. 건너뜁니다.")
                    continue
                if not os.path.exists(os.path.join(DATASET_DIR, edge_file)):
                    print(f">>> ⚠️ 엣지 파일 없음: {edge_file}. 건너뜁니다.")
                    continue

                # 체크포인트명 설정 (GraphMamba 전용 접두사 'gm' 사용)
                checkpoint_name = f"checkpoint_gm_d{distance}_p{error_rate}_{error_type}.pth"
                best_model_name = f"best_gm_d{distance}_p{error_rate}_{error_type}.pth"

                # 데이터 로드
                try:
                    X_train, y_train = load_data(train_file)
                    X_test, y_test = load_data(test_file)
                    edge_index = load_edges(edge_file)
                except Exception as e:
                    print(f">>> 데이터 로드 중 오류 발생: {e}")
                    continue

                # DataLoader 설정
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
                
                # 🆕 GraphMamba 모델 초기화
                model = GraphMamba(
                    num_nodes=X_train.shape[1], 
                    in_channels=X_train.shape[2], 
                    num_qubits=y_train.shape[1],
                    d_model=GM_D_MODEL,
                    num_layers=GM_NUM_LAYERS,
                    dropout=GM_DROPOUT
                ).to(DEVICE)
                
                # pos_weight 계산 (Gradient Explosion 방지: sqrt 적용)
                pos_weight_val = np.sqrt((1.0 - error_rate) / error_rate)
                pos_weight = torch.tensor([pos_weight_val] * y_train.shape[1]).to(DEVICE)
                
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                
                # 체크포인트 로드
                start_epoch, best_ecr, patience_counter = load_checkpoint(model, optimizer, checkpoint_name)
                best_inf_time = 0.0
                best_acc = 0.0

                print(f">>> 학습 루프 시작 ({start_epoch+1} ~ {MAX_EPOCHS} Epochs)...")
                
                for epoch in range(start_epoch, MAX_EPOCHS):
                    # 1. 학습
                    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, edge_index)
                    
                    # 2. 평가
                    val_loss, val_ecr, val_acc, val_time = evaluate(model, test_loader, criterion, DEVICE, edge_index)
                    
                    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
                          f"Loss(T/V): {train_loss:.4f}/{val_loss:.4f} | "
                          f"ECR: {val_ecr:.2%} | Acc: {val_acc:.2%} | "
                          f"Time: {val_time:.4f}ms | "
                          f"Patience: {patience_counter}/{PATIENCE}")
                    
                    # 3. 로그 저장
                    log_epoch_result(distance, error_rate, error_type, epoch + 1, train_loss, val_loss, val_ecr, val_acc, val_time, LEARNING_RATE)

                    # 4. 체크포인트 및 Early Stopping
                    if val_ecr > best_ecr:
                        best_ecr = val_ecr
                        best_acc = val_acc
                        best_inf_time = val_time
                        patience_counter = 0
                        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, best_model_name))
                        save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, checkpoint_name)
                        print(f"    -> 👑 최고 기록 갱신! (ECR: {best_ecr:.2%})")
                    else:
                        patience_counter += 1
                        save_checkpoint(epoch, model, optimizer, best_ecr, patience_counter, checkpoint_name)
                        if patience_counter >= PATIENCE:
                            print(f"\n>>> 🛑 Early Stopping 발동! (Epoch {epoch+1})")
                            break

                print(f"\n>>> 설정 완료 (d={distance}, p={error_rate}, {error_type}). Best ECR: {best_ecr:.2%}")
                
                # 메모리 정리
                del model, optimizer, train_loader, test_loader, X_train, X_test, edge_index
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()