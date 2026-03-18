import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm.auto import tqdm
import numpy as np
import csv
import time
import requests
import json

# --- 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# --- 커스텀 모듈 임포트 ---
from simulation.common.dataset import QECDataset
from models.cnn import CNN
from models.unet import UNet
from models.gcn import GCN
from models.gcnii import GCNII
from models.gat import GAT
from models.appnp import APPNP
from models.gnn import GNN
from models.graph_transformer import GraphTransformer
from models.graph_mamba import GraphMamba

# ==============================================================================
# ⚙️ config.json 로드
# ==============================================================================
def load_config():
    config_path = os.path.join(current_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ config.json not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config.json: {e}")
        sys.exit(1)

CONFIG = load_config()

# 실험 조건
EXP = CONFIG["experiment"]
DISTANCES = EXP["distances"]
ERROR_RATES = EXP["error_rates"]
ERROR_TYPES = EXP["error_types"]
CODE_TYPES = EXP["code_types"]
MEASUREMENTS = EXP["measurements"]
MEAS_NOISE_MAP = EXP["meas_noise"]
MAX_EPOCHS = EXP["max_epochs"]
PATIENCE = EXP["patience"]
AUTOCAST = EXP["autocast"]

# 경로 (config의 상대경로를 절대경로로)
DATASET_BASE = os.path.join(current_dir, EXP["dataset_dir"])
RESULT_BASE = os.path.join(current_dir, EXP["result_dir"])
WEIGHT_BASE = os.path.join(current_dir, EXP["weight_dir"])

# 모델 설정
MODEL_CONFIGS = CONFIG["models"]

# 디바이스
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# keys.json (Discord Webhook)
# ==============================================================================
def load_webhook_url():
    key_file = os.path.join(parent_dir, "keys.json")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                return json.load(f).get("discord_webhook_url", "")
        except:
            return ""
    return ""

DISCORD_WEBHOOK_URL = load_webhook_url()

# ==============================================================================
# 경로 생성 유틸리티
# ==============================================================================
def get_dataset_dir(code_type: str, measurement: str, data_type: str) -> str:
    """dataset/{code_type}/{measurement}/{graph|image}/"""
    return os.path.join(DATASET_BASE, code_type, measurement, data_type)

def get_weight_dir(code_type: str, measurement: str, model_name: str) -> str:
    """saved_weights/{code_type}/{measurement}/{model_name}/"""
    path = os.path.join(WEIGHT_BASE, code_type, measurement, model_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_result_dir(code_type: str, measurement: str) -> str:
    """test_results/{code_type}/{measurement}/"""
    path = os.path.join(RESULT_BASE, code_type, measurement)
    os.makedirs(path, exist_ok=True)
    return path

def get_dataset_files(data_type: str, d: int, p: float, err_type: str):
    """데이터 파일명 생성"""
    sub_dir = "image" if data_type == "image" else "graph"
    train_file = f"train_d{d}_p{p}_{err_type}.npz"
    test_file = f"test_d{d}_p{p}_{err_type}.npz"
    edge_file = f"edges_d{d}.npy" if data_type == "graph" else None
    return sub_dir, train_file, test_file, edge_file

# ==============================================================================
# 유틸리티 함수
# ==============================================================================
def load_data(dir_path, file_name):
    path = os.path.join(dir_path, file_name)
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data['features'], data['labels']

def send_discord_alert(model_name, code_type, meas, d, p, err_type, ecr, acc, time_ms, best_epoch):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        message = {
            "content": f"✅ **[{model_name}] Training Finished!**",
            "embeds": [{
                "title": f"{code_type}/{meas}: d={d}, p={p}, {err_type}",
                "color": 5763719,
                "fields": [
                    {"name": "Best ECR", "value": f"{ecr:.2f}%", "inline": True},
                    {"name": "Best Acc", "value": f"{acc:.2f}%", "inline": True},
                    {"name": "Best Epoch", "value": f"{best_epoch}", "inline": True},
                    {"name": "Inf. Time", "value": f"{time_ms:.4f} ms", "inline": True}
                ],
                "footer": {"text": f"STL Lab Server | {code_type}/{meas}"}
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=message, timeout=5)
    except:
        pass

def get_model_instance(model_name, config, input_shape, num_qubits):
    params = config["params"]

    if model_name == "CNN":
        return CNN(height=input_shape[1], width=input_shape[2], in_channels=1, num_classes=num_qubits)
    elif model_name == "UNet":
        return UNet(in_ch=1, out_ch=num_qubits, base_filters=params.get("base_filters", 32))
    elif model_name == "GCN":
        return GCN(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                   hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
    elif model_name == "GCNII":
        return GCNII(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                     hidden_dim=params["hidden_dim"], num_layers=params["num_layers"],
                     alpha=params["alpha"], theta=params["theta"], dropout=params["dropout"])
    elif model_name == "GAT":
        return GAT(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                   hidden_dim=params["hidden_dim"], heads=params["heads"],
                   num_layers=params["num_layers"], dropout=params["dropout"])
    elif model_name == "APPNP":
        return APPNP(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                     hidden_dim=params["hidden_dim"], K=params["K"], alpha=params["alpha"])
    elif model_name == "GNN":
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

def save_epoch_result(result_dir, model_name, d, p, err_type, epoch, ecr, acc, inf_time, lr, t_loss, v_loss):
    file_path = os.path.join(result_dir, f"benchmark_{model_name}.csv")
    file_exists = os.path.isfile(file_path)
    columns = ["Distance", "Error_Rate(p)", "Error_Type", "Best_ECR(%)", "Accuracy(%)",
               "Inference_Time(ms)", "Epochs", "Learning_Rate", "Train_Loss", "Val_Loss"]
    try:
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow([d, p, err_type, f"{ecr*100:.2f}", f"{acc*100:.2f}", f"{inf_time:.4f}",
                             epoch, lr, f"{t_loss:.6f}", f"{v_loss:.6f}"])
    except Exception as e:
        print(f"⚠️ CSV Save Error: {e}")

def save_checkpoint(weight_dir, model, optimizer, epoch, ecr, model_name, d, p, err_type):
    filename = f"best_{model_name}_d{d}_p{p}_{err_type}.pth"
    path = os.path.join(weight_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_ecr': ecr
    }, path)

# ==============================================================================
# FastTensorDataLoader (GPU Resident)
# ==============================================================================
class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = [t[indices] for t in self.tensors]
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_len:
            raise StopIteration
        end_idx = min(self.current_idx + self.batch_size, self.dataset_len)
        batch = [t[self.current_idx:end_idx] for t in self.tensors]
        self.current_idx += self.batch_size
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

# ==============================================================================
# 🚀 학습 및 평가
# ==============================================================================
def train_and_evaluate(model, train_loader, test_loader, edge_index, criterion, optimizer,
                       config, d, p, err_type, model_name, result_dir, weight_dir):
    best_stats = {"ecr": 0.0, "acc": 0.0, "time": 0.0, "epoch": 0}
    patience_counter = 0
    is_graph = (config["type"] == "graph")
    use_adj = config["use_adj"]
    lr = config["lr"]

    if use_adj:
        num_nodes = train_loader.tensors[0].shape[1]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        eye = torch.eye(num_nodes, device=DEVICE)
        adj = adj + eye
        adj = (adj > 0).float()

    if AUTOCAST:
        scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch} Train", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            inputs = inputs.float()
            labels = labels.float()

            if AUTOCAST:
                with torch.amp.autocast("cuda"):
                    if use_adj:
                        outputs = model(inputs, adj)
                    elif is_graph:
                        outputs = model(inputs, edge_index)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_adj:
                    outputs = model(inputs, adj)
                elif is_graph:
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
                inputs = inputs.float()
                labels = labels.float()

                start = time.time()
                if use_adj:
                    outputs = model(inputs, adj)
                elif is_graph:
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

        print(f"   [Ep {epoch:02d}] Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"ECR: {ecr:.2%} | Acc: {acc:.2%} | Inf.time: {inf_time:.4f}ms | Patience: {patience_counter}")

        save_epoch_result(result_dir, model_name, d, p, err_type, epoch, ecr, acc, inf_time, lr, avg_train_loss, avg_val_loss)

        if ecr > best_stats["ecr"]:
            best_stats = {"ecr": ecr, "acc": acc, "time": inf_time, "epoch": epoch}
            patience_counter = 0
            save_checkpoint(weight_dir, model, optimizer, epoch, ecr, model_name, d, p, err_type)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("   -> 🛑 Early Stopping")
                break

        torch.cuda.empty_cache()

    return best_stats

# ==============================================================================
# 🏁 메인 실행 루프
# ==============================================================================
def main():
    print("=== 🚀 Stim Simulation Benchmark Started ===")
    print(f"Device: {DEVICE}")
    print(f"Code Types: {CODE_TYPES}")
    print(f"Measurements: {MEASUREMENTS}")
    print(f"Distances: {DISTANCES}, Error Rates: {ERROR_RATES}, Error Types: {ERROR_TYPES}")
    print(f"Max Epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
    print(f"Webhook: {'Enabled' if DISCORD_WEBHOOK_URL else 'Disabled'}")

    for code_type in CODE_TYPES:
        for measurement in MEASUREMENTS:
            meas_noise = MEAS_NOISE_MAP[measurement]
            result_dir = get_result_dir(measurement)

            print(f"\n{'#'*70}")
            print(f"# Code: {code_type} | Measurement: {measurement} (noise={meas_noise})")
            print(f"{'#'*70}")

            for model_name, config in MODEL_CONFIGS.items():
                if not config["enabled"]:
                    continue

                weight_dir = get_weight_dir(measurement, model_name)
                data_type = config["type"]  # "image" or "graph"
                data_dir = get_dataset_dir(code_type, measurement, data_type)

                print(f"\n{'='*60}")
                print(f"▶️  {model_name} | {code_type}/{measurement}")
                print(f"    Data: {data_dir}")
                print(f"    Weights: {weight_dir}")
                print(f"{'='*60}")

                for d in DISTANCES:
                    for p in ERROR_RATES:
                        for err_type in ERROR_TYPES:
                            print(f"\n>>> {model_name} (d={d}, p={p}, {err_type})...")

                            # 1. 데이터 로드
                            train_file = f"train_d{d}_p{p}_{err_type}.npz"
                            test_file = f"test_d{d}_p{p}_{err_type}.npz"
                            X_train, y_train = load_data(data_dir, train_file)
                            X_test, y_test = load_data(data_dir, test_file)

                            if X_train is None:
                                print(f"    ⚠️ Data missing ({train_file}). Skipping.")
                                continue

                            print(f"    ✅ Data loaded: Train {X_train.shape}, Test {X_test.shape}")
                            print(f"       Transferring to {DEVICE}...")

                            X_train = torch.from_numpy(np.copy(X_train)).to(torch.uint8).to(DEVICE)
                            y_train = torch.from_numpy(np.copy(y_train)).to(torch.uint8).to(DEVICE)
                            X_test = torch.from_numpy(np.copy(X_test)).to(torch.uint8).to(DEVICE)
                            y_test = torch.from_numpy(np.copy(y_test)).to(torch.uint8).to(DEVICE)

                            # Edge index (Graph 모델용)
                            edge_index = None
                            if data_type == "graph":
                                edge_file = f"edges_d{d}.npy"
                                edge_path = os.path.join(data_dir, edge_file)
                                if os.path.exists(edge_path):
                                    edge_index = torch.LongTensor(np.load(edge_path)).to(DEVICE)
                                else:
                                    print(f"    ⚠️ Edge file missing ({edge_file}). Skipping.")
                                    continue

                            # 2. DataLoader
                            train_loader = FastTensorDataLoader(X_train, y_train, batch_size=config["batch_size"], shuffle=True)
                            test_loader = FastTensorDataLoader(X_test, y_test, batch_size=config["batch_size"], shuffle=False)

                            # 3. 모델 초기화
                            input_shape = X_train.shape[1:]
                            num_qubits = y_train.shape[1]

                            model = get_model_instance(model_name, config, input_shape, num_qubits)
                            if model is None:
                                continue
                            model = model.to(DEVICE)

                            # 4. Loss & Optimizer
                            if config["loss_weight"] == "sqrt":
                                w_val = np.sqrt((1.0 - p) / p)
                            else:
                                w_val = (1.0 - p) / p

                            pos_weight = torch.tensor([w_val] * num_qubits).to(DEVICE)
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            optimizer = optim.Adam(model.parameters(), lr=config["lr"])

                            # 5. 학습
                            try:
                                best_stats = train_and_evaluate(
                                    model, train_loader, test_loader, edge_index,
                                    criterion, optimizer, config, d, p, err_type, model_name,
                                    result_dir, weight_dir
                                )
                                send_discord_alert(model_name, code_type, measurement,
                                                   d, p, err_type,
                                                   best_stats["ecr"]*100, best_stats["acc"]*100,
                                                   best_stats["time"], best_stats["epoch"])
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    print(f"    ❌ OOM Error! Skipping.")
                                    torch.cuda.empty_cache()
                                else:
                                    print(f"    ❌ Runtime Error: {e}")

                            # 메모리 정리
                            del model, optimizer, train_loader, test_loader, X_train, X_test
                            torch.cuda.empty_cache()

    print("\n=== ✨ All Benchmarks Completed! ===")

if __name__ == "__main__":
    main()
