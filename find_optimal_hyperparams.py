import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from simulation.common.dataset import QECDataset
from models.graph_transformer import GraphTransformer
from utils.focal_loss import FocalLoss

DISTANCE = 3
ERROR_RATE = 0.05
ERROR_TYPE = "Z"
DATASET_DIR = "dataset/color_code/graph"
TRAIN_FILE = f"train_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"
TEST_FILE = f"test_d{DISTANCE}_p{ERROR_RATE}_{ERROR_TYPE}.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_SAMPLES = 10000  # Quick test
EPOCHS_PER_CONFIG = 5

def load_data(file_name):
    path = os.path.join(DATASET_DIR, file_name)
    data = np.load(path)
    features, labels = data['features'], data['labels']
    
    # Subsample for quick testing
    if len(features) > MAX_SAMPLES:
        indices = np.random.permutation(len(features))[:MAX_SAMPLES]
        features = features[indices]
        labels = labels[indices]
    
    return features, labels

def evaluate_with_threshold(model, loader, device):
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    best_threshold = 0.0
    best_f1 = 0.0
    best_ecr = 0.0
    best_acc = 0.0
    
    for threshold in np.arange(-3.0, 3.0, 0.5):
        preds = (all_outputs > threshold).float()
        
        tp = ((preds == 1) & (all_labels == 1)).sum().item()
        fp = ((preds == 1) & (all_labels == 0)).sum().item()
        fn = ((preds == 0) & (all_labels == 1)).sum().item()
        tn = ((preds == 0) & (all_labels == 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        ecr = recall  # ECR is essentially recall for error detection
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_ecr = ecr
            best_acc = accuracy
    
    return best_f1, best_ecr, best_acc, best_threshold

def test_config(alpha, gamma, lr, X_train, y_train, X_test, y_test):
    train_loader = DataLoader(QECDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(QECDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = GraphTransformer(
        num_nodes=X_train.shape[1],
        in_channels=X_train.shape[2],
        num_qubits=y_train.shape[1],
        d_model=128,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    ).to(DEVICE)
    
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Quick training
    for epoch in range(EPOCHS_PER_CONFIG):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Evaluate
    f1, ecr, acc, threshold = evaluate_with_threshold(model, test_loader, DEVICE)
    
    return f1, ecr, acc, threshold

def main():
    print("=== Hyperparameter Search ===")
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Grid search
    alphas = [0.25, 0.5, 0.75, 1.0]
    gammas = [1.0, 2.0, 3.0]
    lrs = [1e-3, 2e-3]
    
    results = []
    
    for alpha in alphas:
        for gamma in gammas:
            for lr in lrs:
                print(f"\nTesting: alpha={alpha}, gamma={gamma}, lr={lr}")
                try:
                    f1, ecr, acc, threshold = test_config(alpha, gamma, lr, X_train, y_train, X_test, y_test)
                    results.append({
                        'alpha': alpha,
                        'gamma': gamma,
                        'lr': lr,
                        'f1': f1,
                        'ecr': ecr,
                        'acc': acc,
                        'threshold': threshold
                    })
                    print(f"  → F1: {f1:.4f} | ECR: {ecr:.2%} | Acc: {acc:.2%} | Threshold: {threshold:.2f}")
                except Exception as e:
                    print(f"  → Failed: {e}")
    
    # Sort by F1 score
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS:")
    print("="*60)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. alpha={r['alpha']}, gamma={r['gamma']}, lr={r['lr']}")
        print(f"   F1: {r['f1']:.4f} | ECR: {r['ecr']:.2%} | Acc: {r['acc']:.2%} | Threshold: {r['threshold']:.2f}")
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION:")
    best = results[0]
    print(f"alpha={best['alpha']}, gamma={best['gamma']}, lr={best['lr']}")
    print(f"F1: {best['f1']:.4f} | ECR: {best['ecr']:.2%} | Acc: {best['acc']:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()
