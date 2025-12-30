import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import csv
from tqdm import tqdm

from models.umamba import UMamba
from models.unet import UNet

# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["UMamba"] # "UMamba" or "UNet"
DISTANCES = [3, 5, 7]
PROBS = [0.005, 0.01, 0.05]
EPOCH = 10
DATA_DIR = "dataset/surface_code"
RESULT_DIR = "test_results"
WEIGHT_DIR = "saved_weights"
LOSS_DIR = "saved_weights/loss_logs"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)

# Utils
class SurfaceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.FloatTensor(data['X'])
        self.y = torch.FloatTensor(data['y'])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def calculate_detailed_metrics(pred_logits, target):
    probs = torch.sigmoid(pred_logits)
    preds = (probs > 0.5).float()
    correct_pixels = (preds == target).sum().item()
    total_pixels = target.numel()
    accuracy = correct_pixels / total_pixels
    target_errors = (target == 1)
    predicted_errors = (preds == 1)
    tp = (target_errors & predicted_errors).sum().item()
    fn = (target_errors & ~predicted_errors).sum().item()
    if (tp + fn) > 0:
        ecr = tp / (tp + fn)
    else:
        ecr = 1.0 
    return accuracy, ecr

# __main__
for MODEL in MODELS:
    for DISTANCE in DISTANCES:
        for PROB in PROBS:
            TRAIN_PATH = f"{DATA_DIR}/surface_train_d{DISTANCE}_p{PROB}.npz"
            TEST_PATH = f"{DATA_DIR}/surface_test_d{DISTANCE}_p{PROB}.npz"
            
            model_weight_dir = f"{WEIGHT_DIR}/surface_code/{MODEL}"
            os.makedirs(model_weight_dir, exist_ok=True)
            WEIGHT_PATH = f"{model_weight_dir}/{MODEL}_d{DISTANCE}_p{PROB}.pth"
            LOSS_LOG_PATH = f"{LOSS_DIR}/loss_log_{MODEL}_d{DISTANCE}_p{PROB}.csv"
            
            if not os.path.exists(TRAIN_PATH):
                print(f"데이터 파일이 없습니다: {TRAIN_PATH}")
                continue 
            
            # Load Data
            train_dataset = SurfaceDataset(TRAIN_PATH)
            test_dataset = SurfaceDataset(TEST_PATH)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize Model
            if MODEL == "UMamba":
                model = UMamba(in_ch=2, out_ch=2).to(device)
            else:
                model = UNet(in_ch=2, out_ch=2).to(device)
            
            previous_epochs = 0
            skip_training = False 

            # Load Weights
            if os.path.exists(WEIGHT_PATH):
                print(f">>> Loading checkpoint from {WEIGHT_PATH}")
                checkpoint = torch.load(WEIGHT_PATH)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    previous_epochs = checkpoint.get('epoch', 0)
                else:
                    model.load_state_dict(checkpoint)
                    previous_epochs = 0 
                print(f"    (Previously trained for {previous_epochs} epochs)")
                skip_training = False 
            else:
                print(">>> No pre-trained weights found. Starting fresh training...")
                previous_epochs = 0
                skip_training = False
                    
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.BCEWithLogitsLoss()
            
            # --- Training Loop ---
            if not skip_training:
                print(f">>> Start Training {MODEL} (d={DISTANCE}, p={PROB})")
                model.train()
                
                # [NEW] Loss History 저장을 위한 파일 열기 (append mode)
                loss_file_exists = os.path.exists(LOSS_LOG_PATH)
                
                with open(LOSS_LOG_PATH, mode='a', newline='') as f_loss:
                    writer_loss = csv.writer(f_loss)
                    if not loss_file_exists:
                        writer_loss.writerow(['Epoch', 'Loss']) # Header
                    
                    for epoch in range(EPOCH):
                        epoch_loss = 0
                        current_real_epoch = previous_epochs + epoch + 1
                        
                        for X, y in tqdm(train_loader, desc=f"Epoch {current_real_epoch}"):
                            X, y = X.to(device), y.to(device)
                            optimizer.zero_grad()
                            outputs = model(X)
                            loss = criterion(outputs, y)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                        
                        avg_loss = epoch_loss / len(train_loader)
                        print(f"Epoch {current_real_epoch} Loss: {avg_loss:.5f}")
                        
                        # [NEW] Epoch 끝날 때마다 Loss 기록
                        writer_loss.writerow([current_real_epoch, f"{avg_loss:.6f}"])
                        f_loss.flush() # 즉시 파일에 쓰기 (중간에 꺼져도 저장되게)

                total_trained_epochs = previous_epochs + EPOCH
            else:
                total_trained_epochs = previous_epochs

            # --- Final Benchmark Evaluation ---
            print("\n>>> Running Final Benchmark on Test Set...")
            model.eval()
            total_acc = 0
            total_ecr = 0
            total_loss = 0
            
            start_time = time.time()
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    total_loss += loss.item()
                    acc, ecr = calculate_detailed_metrics(outputs, y)
                    total_acc += acc
                    total_ecr += ecr
                    
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000 / len(test_dataset)
            
            final_loss = total_loss / len(test_loader)
            final_acc = (total_acc / len(test_loader)) * 100
            final_ecr = (total_ecr / len(test_loader)) * 100
            
            print(f"=== Final Results for {MODEL} (Total Epochs: {total_trained_epochs}) ===")
            print(f"Loss: {final_loss:.4f}")
            print(f"Accuracy: {final_acc:.2f}%")
            print(f"ECR: {final_ecr:.2f}%")
            print(f"Inference Time: {inference_time_ms:.4f} ms/sample")
            
            torch.save({
                'epoch': total_trained_epochs,
                'model_state_dict': model.state_dict(),
                'loss': final_loss
            }, WEIGHT_PATH)
            print(f">>> Checkpoint saved to {WEIGHT_PATH}")
            
            csv_file = f"{RESULT_DIR}/benchmark_surface.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Model', 'Distance', 'Prob', 'Accuracy', 'ECR', 'Inference_Time_ms', 'Loss', 'Total_Epochs'])
                writer.writerow([MODEL, DISTANCE, PROB, f"{final_acc:.4f}", f"{final_ecr:.4f}", f"{inference_time_ms:.4f}", f"{final_loss:.6f}", total_trained_epochs])
                
            print(f">>> Results saved to {csv_file}\n")