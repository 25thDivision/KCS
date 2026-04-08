"""
Stim Simulation Benchmark

사용법:
  python3 stim_simulation/run_stim_simulation.py
  python3 stim_simulation/run_stim_simulation.py -g 0 -m GraphMamba GraphTransformer -c color_code -d 3
  python3 stim_simulation/run_stim_simulation.py -g 1 -m CNN GCN -n realistic/dp0_mf0.005_rf0.005_gd0.004
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import csv
import time
import requests
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(root_dir)

from models.cnn import CNN
from models.unet import UNet
from models.gcn import GCN
from models.gcnii import GCNII
from models.gat import GAT
from models.appnp import APPNP
from models.gnn import GNN
from models.graph_transformer import GraphTransformer
from models.graph_mamba import GraphMamba
from utils.focal_loss import FocalLoss, CombinedLoss
from logger import log_to_file
from paths import ProjectPaths

# ==============================================================================
# CLI
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stim Simulation Benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("-g", "--gpu", type=int, default=None)
    parser.add_argument("-m", "--models", nargs="+", type=str, default=None)
    parser.add_argument("-c", "--code", nargs="+", type=str, default=None)
    parser.add_argument("-n", "--noise", nargs="+", type=str, default=None)
    parser.add_argument("-d", "--distance", nargs="+", type=int, default=None)
    return parser.parse_args()

ARGS = parse_args()

if ARGS.gpu is not None:
    torch.cuda.set_device(ARGS.gpu)
    DEVICE = f"cuda:{ARGS.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATHS = ProjectPaths(root_dir)

def load_config():
    config_path = ARGS.config if ARGS.config else PATHS.stim_config()
    if not os.path.exists(config_path):
        print(f"⚠️ config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()
EXP = CONFIG["experiment"]

NOISE_PROFILES = EXP["noise_profiles"]
ERROR_RATES = EXP["error_rates"]
ERROR_TYPES = EXP["error_types"]
MAX_EPOCHS = EXP["max_epochs"]
PATIENCE = EXP["patience"]
AUTOCAST = EXP["autocast"]
MODEL_CONFIGS = CONFIG["models"]

CODE_TYPES = ARGS.code if ARGS.code else EXP["code_types"]
ACTIVE_NOISE = ARGS.noise if ARGS.noise else EXP["active_noise"]
DISTANCES = ARGS.distance if ARGS.distance else EXP["distances"]

KEYS = PATHS.load_keys()
DISCORD_WEBHOOK_URL = KEYS.get("discord_simulation", "")

def send_discord_alert(model_name, code_type, noise, d, p, err_type, ecr, acc, time_ms, best_epoch):
    log_to_file(f"{model_name} | {code_type}/{noise} | d={d}, p={p}, {err_type} | ECR={ecr:.2f}% | Acc={acc:.2f}%")
    
    if not DISCORD_WEBHOOK_URL:
        return
    
    try:
        noise_type, noise_params = noise.split('/')
        noise_type = noise_type.capitalize() 
    except ValueError:
        noise_type = noise.capitalize()
        noise_params = "N/A"
    
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={
            "content": f"✅ **[{model_name}] Training Finished!**",
            "embeds": [{
                "title": f"📊 {code_type} Experiment Results",  # 제목은 심플하게 모델/코드명만!
                "description": f"**Noise**: `{noise_type}` (`{noise_params}`)\n**Setting**: `d={d}`, `p={p}`, Error: `{err_type}`", # 긴 설정값은 description으로!
                "color": 5763719,
                "fields": [
                    # 결과값들 (한 줄에 3개 배치)
                    {"name": "🏆 Best ECR", "value": f"{ecr:.2f}%", "inline": True},
                    {"name": "🎯 Best Acc", "value": f"{acc:.2f}%", "inline": True},
                    {"name": "⏱️ Inf. Time", "value": f"{time_ms:.4f} ms", "inline": True},
                    
                    # 4번째 필드는 다음 줄에 표시됨
                    {"name": "📌 Best Epoch", "value": f"Epoch {best_epoch}", "inline": True},
                ],
                "footer": {"text": f"STL Lab Server | GPU: {ARGS.gpu if ARGS.gpu is not None else 'auto'}"}
            }]
        }, timeout=5)
    except:
        log_to_file(f"Failed to send Discord alert: {model_name} | {code_type}/{noise} | d={d}, p={p}, {err_type} | ECR={ecr:.2f}% | Acc={acc:.2f}%")
        pass

# ==============================================================================
# 유틸리티
# ==============================================================================
def load_data(filepath):
    if not os.path.exists(filepath):
        return None, None
    data = np.load(filepath)
    return data['features'], data['labels']

def get_model_instance(model_name, config, input_shape, num_qubits):
    params = config["params"]
    if model_name == "CNN":
        return CNN(height=input_shape[1], width=input_shape[2], in_channels=input_shape[0], num_classes=num_qubits)
    elif model_name == "UNet":
        return UNet(in_ch=input_shape[0], out_ch=num_qubits, base_filters=params.get("base_filters", 32))
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

def save_epoch_result(filepath, model_name, d, p, err_type, epoch, ecr, acc, inf_time, lr, t_loss, v_loss):
    file_exists = os.path.isfile(filepath)
    columns = ["Distance", "Error_Rate(p)", "Error_Type", "Best_ECR(%)", "Accuracy(%)",
               "Inference_Time(ms)", "Epochs", "Learning_Rate", "Train_Loss", "Val_Loss"]
    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow([d, p, err_type, f"{ecr*100:.2f}", f"{acc*100:.2f}", f"{inf_time:.4f}",
                             epoch, lr, f"{t_loss:.6f}", f"{v_loss:.6f}"])
    except Exception as e:
        print(f"⚠️ CSV Save Error: {e}")

def save_checkpoint(filepath, model, optimizer, epoch, ecr):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch, 'best_ecr': ecr}, filepath)

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
# 학습
# ==============================================================================
def train_and_evaluate(model, train_loader, test_loader, edge_index, criterion, optimizer,
                       config, d, p, err_type, model_name, result_filepath, weight_filepath):
    best_stats = {"ecr": 0.0, "acc": 0.0, "time": 0.0, "epoch": 0}
    patience_counter = 0
    is_graph = (config["type"] == "graph")
    use_adj = config["use_adj"]
    lr = config["lr"]

    if use_adj:
        from torch_geometric.utils import to_dense_adj
        num_nodes = train_loader.tensors[0].shape[1]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        adj = adj + torch.eye(num_nodes, device=DEVICE)
        adj = (adj > 0).float()

    if AUTOCAST:
        scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch} Train", leave=False):
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE).float()
            optimizer.zero_grad()
            if AUTOCAST:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs, adj) if use_adj else (model(inputs, edge_index) if is_graph else model(inputs))
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                outputs = model(inputs, adj) if use_adj else (model(inputs, edge_index) if is_graph else model(inputs))
                loss = criterion(outputs, labels)
                loss.backward(); optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = total_error_bits = detected_error_bits = correct_bits = total_bits = 0
        total_time = total_samples = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.float(), labels.float()
                start = time.time()
                outputs = model(inputs, adj) if use_adj else (model(inputs, edge_index) if is_graph else model(inputs))
                total_time += time.time() - start
                total_samples += inputs.size(0)
                val_loss += criterion(outputs, labels).item()
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
              f"ECR: {ecr:.2%} | Acc: {acc:.2%} | Inf: {inf_time:.4f}ms | Pat: {patience_counter}")

        save_epoch_result(result_filepath, model_name, d, p, err_type, epoch, ecr, acc, inf_time, lr, avg_train_loss, avg_val_loss)

        if ecr > best_stats["ecr"]:
            best_stats = {"ecr": ecr, "acc": acc, "time": inf_time, "epoch": epoch}
            patience_counter = 0
            save_checkpoint(weight_filepath, model, optimizer, epoch, ecr)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("   -> 🛑 Early Stopping")
                break
        torch.cuda.empty_cache()
    return best_stats

# ==============================================================================
# 메인
# ==============================================================================
def main():
    print("=== 🚀 Stim Simulation Benchmark ===")
    print(f"Device: {DEVICE} | GPU: {ARGS.gpu or 'auto'}")
    print(f"Models: {ARGS.models or 'all enabled'}")
    print(f"Code: {CODE_TYPES} | Noise: {ACTIVE_NOISE} | Dist: {DISTANCES}")
    print(f"Error Rates: {ERROR_RATES} | Error Types: {ERROR_TYPES}")
    print(f"Epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")

    for code_type in CODE_TYPES:
        for noise in ACTIVE_NOISE:
            if noise not in NOISE_PROFILES:
                print(f"⚠️ Noise '{noise}' not in config. Skipping.")
                continue

            print(f"\n{'#'*70}\n# {code_type}/{noise}\n# Params: {NOISE_PROFILES[noise]}\n{'#'*70}")

            for model_name, config in MODEL_CONFIGS.items():
                if not config["enabled"]:
                    continue
                if ARGS.models and model_name not in ARGS.models:
                    continue

                data_type = config["type"]
                print(f"\n{'='*60}\n▶️  {model_name} | {code_type}/{noise}\n{'='*60}")

                for d in DISTANCES:
                    for p in ERROR_RATES:
                        for err_type in ERROR_TYPES:
                            print(f"\n>>> {model_name} (d={d}, p={p}, {err_type})...")

                            train_path = PATHS.stim_data(code_type, noise, data_type, "train", d, p, err_type)
                            test_path = PATHS.stim_data(code_type, noise, data_type, "test", d, p, err_type)
                            X_train, y_train = load_data(train_path)
                            X_test, y_test = load_data(test_path)

                            if X_train is None:
                                print(f"    ⚠️ Data missing: {train_path}. Skipping.")
                                continue

                            print(f"    ✅ Train {X_train.shape}, Test {X_test.shape} → {DEVICE}")
                            X_train = torch.from_numpy(np.copy(X_train)).to(torch.uint8).to(DEVICE)
                            y_train = torch.from_numpy(np.copy(y_train)).to(torch.uint8).to(DEVICE)
                            X_test = torch.from_numpy(np.copy(X_test)).to(torch.uint8).to(DEVICE)
                            y_test = torch.from_numpy(np.copy(y_test)).to(torch.uint8).to(DEVICE)

                            edge_index = None
                            if data_type == "graph":
                                edge_path = PATHS.stim_edge(code_type, noise, d)
                                if os.path.exists(edge_path):
                                    edge_index = torch.LongTensor(np.load(edge_path)).to(DEVICE)
                                else:
                                    print(f"    ⚠️ Edge missing: {edge_path}. Skipping.")
                                    continue

                            train_loader = FastTensorDataLoader(X_train, y_train, batch_size=config["batch_size"], shuffle=True)
                            test_loader = FastTensorDataLoader(X_test, y_test, batch_size=config["batch_size"], shuffle=False)

                            model = get_model_instance(model_name, config, X_train.shape[1:], y_train.shape[1])
                            if model is None:
                                continue
                            model = model.to(DEVICE)

                            loss_type = config.get("loss_type", "bce")
                            w_val = np.sqrt((1.0 - p) / p) if config["loss_weight"] == "sqrt" else (1.0 - p) / p
                            pos_weight = torch.tensor([w_val] * y_train.shape[1]).to(DEVICE)
                            if loss_type == "focal":
                                criterion = FocalLoss(alpha=0.25, gamma=2.0)
                            elif loss_type == "combined":
                                criterion = CombinedLoss(pos_weight=pos_weight, diversity_weight=0.1)
                            else:
                                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            
                            optimizer = optim.Adam(model.parameters(), lr=config["lr"])

                            result_fp = PATHS.stim_result(code_type, noise, model_name)
                            weight_fp = PATHS.stim_weight(code_type, noise, model_name, d, p, err_type)

                            try:
                                best = train_and_evaluate(model, train_loader, test_loader, edge_index,
                                    criterion, optimizer, config, d, p, err_type, model_name, result_fp, weight_fp)
                                send_discord_alert(model_name, code_type, noise, d, p, err_type,
                                    best["ecr"]*100, best["acc"]*100, best["time"], best["epoch"])
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    print(f"    ❌ OOM! Skipping.")
                                    torch.cuda.empty_cache()
                                else:
                                    print(f"    ❌ Error: {e}")

                            del model, optimizer, train_loader, test_loader, X_train, X_test
                            torch.cuda.empty_cache()

    print("\n=== ✨ All Benchmarks Completed! ===")

if __name__ == "__main__":
    main()