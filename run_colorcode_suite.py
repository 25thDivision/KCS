import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import csv
import time
import requests
import json

# --- 모듈 경로 설정 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 커스텀 모듈 임포트 ---
from simulation.common.dataset import QECDataset
from models.cnn import CNN
from models.gnn import GNN
from models.unet import UNet
from models.graph_transformer import GraphTransformer
try:
    from models.graph_mamba import GraphMamba
except ImportError:
    GraphMamba = None
    print("⚠️ GraphMamba(mamba-ssm) is not installed. Skipping related models.")

# ==============================================================================
# ⚙️ [User Config] 하이퍼파라미터 & 설정 관리
# ==============================================================================

# 1. 실행할 모델 및 하이퍼파라미터 정의
MODEL_CONFIGS = {
    "CNN": {
        "enabled": True,
        "type": "image",
        "batch_size": 256,
        "lr": 1e-3,
        "loss_weight": "linear", # (1-p)/p
        "params": {}
    },
    "UNet": {
        "enabled": False,  # False임! 확인!
        "type": "image",
        "batch_size": 128,
        "lr": 1e-3,
        "loss_weight": "linear",
        "params": {
            "base_filters": 32
        }
    },
    "GNN": {
        "enabled": True,
        "type": "graph",
        "batch_size": 512,
        "lr": 1e-3,
        "loss_weight": "linear",
        "params": {
            "hidden_dim": 64,
            "num_layers": 4
        }
    },
    "GraphTransformer": {
        "enabled": True,
        "type": "graph",
        "batch_size": 256, 
        "lr": 1e-4,
        "loss_weight": "sqrt", # sqrt((1-p)/p)
        "params": {
            "d_model": 256,
            "num_heads": 4,
            "num_layers": 5,
            "dropout": 0.1
        }
    },
    "GraphMamba": {
        "enabled": False if GraphMamba else False,
        "type": "graph",
        "batch_size": 256,
        "lr": 1e-4,
        "loss_weight": "sqrt",
        "params": {
            "d_model": 64,
            "num_layers": 4,
            "dropout": 0.1
        }
    }
}

# 2. 실험 조건
DISTANCES = [3, 5, 7]
ERROR_RATES = [0.005, 0.01, 0.05]
ERROR_TYPES = ["X", "Z"]
# 2-1. 실험 환경
NUM_WORKERS = min(16, os.cpu_count() - 2) if os.cpu_count() else 0

# 3. 경로 설정
BASE_DATA_DIR = "dataset/color_code"
RESULT_DIR = "test_results/server"
WEIGHT_DIR = "saved_weights/server"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)

# 4. 공통 학습 설정
MAX_EPOCHS = 20
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 5. 디스코드 웹후크 설정



# ==============================================================================
# 🛠️ 유틸리티 함수
# ==============================================================================

def send_discord_alert(model_name, d, p, err_type, ecr, acc, time_ms, best_epoch):
    if not DISCORD_WEBHOOK_URL: return
    
    try:
        message = {
            "content": f"✅ **[{model_name}] Training Finished!**",
            "embeds": [{
                "title": f"Config: d={d}, p={p}, {err_type}",
                "color": 5763719, # Green
                "fields": [
                    {"name": "Best ECR", "value": f"{ecr:.2f}%", "inline": True},
                    {"name": "Best Acc", "value": f"{acc:.2f}%", "inline": True},
                    {"name": "Best Epoch", "value": f"{best_epoch}", "inline": True},
                    {"name": "Inf. Time", "value": f"{time_ms:.4f} ms", "inline": True}
                ],
                "footer": {"text": "My Lab Server"}
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=message, timeout=5)
    except Exception as e:
        print(f"⚠️ Discord Webhook Error: {e}")

def get_dataset_paths(data_type, d, p, err_type):
    sub_dir = "image" if data_type == "image" else "graph"
    data_dir = os.path.join(BASE_DATA_DIR, sub_dir)
    train_file = f"train_d{d}_p{p}_{err_type}.npz"
    test_file = f"test_d{d}_p{p}_{err_type}.npz"
    edge_file = f"edges_d{d}.npy" if data_type == "graph" else None
    return data_dir, train_file, test_file, edge_file

def load_data(dir_path, file_name):
    path = os.path.join(dir_path, file_name)
    if not os.path.exists(path): return None, None
    data = np.load(path)
    return data['features'], data['labels']

def get_model_instance(model_name, config, input_shape, num_qubits):
    params = config["params"]
    
    if model_name == "CNN":
        # input: (1, H, W)
        return CNN(height=input_shape[1], width=input_shape[2], in_channels=1, num_classes=num_qubits)
    
    elif model_name == "UNet":
        # UNet은 보통 (N, C, H, W) 입력을 받아 (N, OutC, H, W) 혹은 Flatten 출력을 내보냄
        # models/unet.py의 구현에 따라 초기화 인자가 다를 수 있음 (일반적인 파라미터 적용)
        return UNet(in_ch=1, out_ch=num_qubits, base_filters=params.get("base_filters", 32))
    
    elif model_name == "GNN":
        # input: (Nodes, Feats)
        return GNN(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits, 
                   hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
    
    elif model_name == "GraphTransformer":
        return GraphTransformer(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits, 
                                d_model=params["d_model"], num_heads=params["num_heads"], 
                                num_layers=params["num_layers"], dropout=params["dropout"])
    
    elif model_name == "GraphMamba":
        return GraphMamba(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits, 
                          d_model=params["d_model"], num_layers=params["num_layers"], dropout=params["dropout"])
    return None

def save_epoch_result(model_name, d, p, err_type, epoch, ecr, acc, inf_time, lr, t_loss, v_loss):
    """매 Epoch 결과를 CSV에 누적 저장"""
    file_path = os.path.join(RESULT_DIR, f"benchmark_results_server_{model_name}.csv")
    file_exists = os.path.isfile(file_path)
    columns = ["Distance", "Error_Rate(p)", "Error_Type", "Best_ECR(%)", "Accuracy(%)", 
               "Inference_Time(ms)", "Epochs", "Learning_Rate", "Train_Loss", "Val_Loss"]
    
    try:
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(columns)
            writer.writerow([d, p, err_type, f"{ecr*100:.2f}", f"{acc*100:.2f}", f"{inf_time:.4f}",
                             epoch, lr, f"{t_loss:.6f}", f"{v_loss:.6f}"])
    except Exception as e:
        print(f"⚠️ CSV Save Error: {e}")

def save_checkpoint(model, optimizer, epoch, ecr, model_name, d, p, err_type):
    """최고 성능 모델 저장"""
    filename = f"best_{model_name}_d{d}_p{p}_{err_type}.pth"
    path = os.path.join(WEIGHT_DIR, filename)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_ecr': ecr
    }
    torch.save(state, path)

# ==============================================================================
# 🚀 학습 및 평가 코어 함수
# ==============================================================================
def train_and_evaluate(model, train_loader, test_loader, edge_index, criterion, optimizer, config, d, p, err_type, model_name):
    best_stats = {"ecr": 0.0, "acc": 0.0, "time": 0.0, "epoch": 0}
    patience_counter = 0
    is_graph = (config["type"] == "graph")
    lr = config["lr"]

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        # tqdm progress bar (leave=False: 완료 시 사라짐 -> 로그 깔끔하게 유지)
        pbar = tqdm(train_loader, desc=f"Ep {epoch} Train", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if is_graph:
                outputs = model(inputs, edge_index)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        val_loss = 0.0
        total_error_bits = 0
        detected_error_bits = 0
        correct_bits = 0
        total_bits = 0
        total_time = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                start = time.time()
                if is_graph:
                    outputs = model(inputs, edge_index)
                else:
                    outputs = model(inputs)
                end = time.time()
                
                total_time += (end - start)
                total_samples += inputs.size(0)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0).float()
                error_mask = (labels == 1)
                total_error_bits += error_mask.sum().item()
                detected_error_bits += (preds[error_mask] == 1).sum().item()
                correct_bits += (preds == labels).sum().item()
                total_bits += labels.numel()

        avg_val_loss = val_loss / len(test_loader)
        ecr = detected_error_bits / total_error_bits if total_error_bits > 0 else 0.0
        acc = correct_bits / total_bits
        inf_time = (total_time / total_samples) * 1000

        # 로그 출력 (한 줄로 깔끔하게)
        print(f"   [Ep {epoch:02d}] Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | ECR: {ecr:.2%} | Acc: {acc:.2%} | Inf.time: {inf_time:.4f}ms | Patience: {patience_counter}")

        # CSV 저장
        save_epoch_result(model_name, d, p, err_type, epoch, ecr, acc, inf_time, lr, avg_train_loss, avg_val_loss)

        # Best Model 갱신 & 체크포인트 저장
        if ecr > best_stats["ecr"]:
            best_stats = {"ecr": ecr, "acc": acc, "time": inf_time, "epoch": epoch}
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, ecr, model_name, d, p, err_type)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("   -> 🛑 Early Stopping")
                break
                
    return best_stats

# ==============================================================================
# 🏁 메인 실행 루프
# ==============================================================================
def main():
    print("=== 🚀 Integrated Benchmark Server Started ===")
    print(f"Directory: {RESULT_DIR}")
    print(f"Workers: {NUM_WORKERS}, Device: {DEVICE}")
    print(f"Webhook: {'Enabled' if DISCORD_WEBHOOK_URL else 'Disabled'}")

    for model_name, config in MODEL_CONFIGS.items():
        if not config["enabled"]: continue
            
        print(f"\n{'='*60}")
        print(f"▶️  Starting Benchmark for: {model_name}")
        print(f"    Params: {config['params']}")
        print(f"{'='*60}")
        
        for d in DISTANCES:
            for p in ERROR_RATES:
                for err_type in ERROR_TYPES:
                    print(f"\n>>> Running {model_name} (d={d}, p={p}, {err_type})...")
                    
                    # 1. 데이터 로드
                    data_dir, train_f, test_f, edge_f = get_dataset_paths(config["type"], d, p, err_type)
                    X_train, y_train = load_data(data_dir, train_f)
                    X_test, y_test = load_data(data_dir, test_f)
                    
                    if X_train is None:
                        print(f"    ⚠️ Data missing ({train_f}). Skipping.")
                        continue
                    
                    edge_index = None
                    if config["type"] == "graph":
                        edge_path = os.path.join(data_dir, edge_f)
                        if os.path.exists(edge_path):
                            edge_index = torch.LongTensor(np.load(edge_path)).to(DEVICE)
                        else:
                            print(f"    ⚠️ Edge file missing ({edge_f}). Skipping.")
                            continue

                    # 2. DataLoader (pin_memory=True, NUM_WORKERS 적용)
                    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=config["batch_size"], 
                                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
                    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=config["batch_size"], 
                                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
                    
                    # 3. 모델 초기화
                    input_shape = X_train.shape[1:] 
                    num_qubits = y_train.shape[1]
                    
                    model = get_model_instance(model_name, config, input_shape, num_qubits)
                    if model is None: continue
                    model = model.to(DEVICE)
                    
                    # 4. Loss Weighting 설정
                    if config["loss_weight"] == "sqrt":
                        w_val = np.sqrt((1.0 - p) / p)
                    else: # linear
                        w_val = (1.0 - p) / p
                        
                    pos_weight = torch.tensor([w_val] * num_qubits).to(DEVICE)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
                    
                    # 5. 학습 실행
                    try:
                        best_stats = train_and_evaluate(
                            model, train_loader, test_loader, edge_index, 
                            criterion, optimizer, config, d, p, err_type, model_name
                        )
                        
                        # 6. 알림 전송
                        send_discord_alert(model_name, d, p, err_type, 
                                           best_stats["ecr"]*100, best_stats["acc"]*100, 
                                           best_stats["time"], best_stats["epoch"])
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"    ❌ OOM Error! Skipping this config. (Try lowering batch_size)")
                            torch.cuda.empty_cache()
                        else:
                            print(f"    ❌ Runtime Error: {e}")
                    
                    # 메모리 정리
                    del model, optimizer, train_loader, test_loader, X_train, X_test
                    torch.cuda.empty_cache()

    print("\n=== ✨ All Benchmarks Completed Successfully! ===")

if __name__ == "__main__":
    main()