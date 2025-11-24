import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ----------------------------------------------------
# 0. Dataset 클래스 (새로 추가됨)
# ----------------------------------------------------
class QECDataset(Dataset):
    """
    Numpy 데이터를 PyTorch 학습용 데이터셋으로 변환하는 클래스입니다
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, 1, H, W) for CNN or (N, Nodes, Feats) for GNN
            labels: (N, Num_Qubits) 물리적 에러 라벨
        """
        # 모델 입력과 손실 함수 계산을 위해 FloatTensor로 변환합니다
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----------------------------------------------------
# 1. 손실 함수 (Loss Function) 정의
# ----------------------------------------------------
# Multi-label classification(각 큐비트가 에러인지 아닌지 독립적으로 판단)에 적합합니다
loss_fn = nn.BCEWithLogitsLoss()


# ----------------------------------------------------
# 2. 1 에포크(Epoch) 학습 함수
# ----------------------------------------------------
def train_one_epoch(model, data_loader, optimizer, device='cuda'):
    """
    모델을 1 에포크 동안 학습시킵니다
    """
    model.train() # 학습 모드
    model.to(device)
    
    total_loss = 0.0
    
    for inputs, labels in tqdm(data_loader, desc="Training Epoch", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(inputs)
        
        # 손실 계산 (Output shape: [Batch, Num_Qubits], Label shape: [Batch, Num_Qubits])
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)


# ----------------------------------------------------
# 3. 평가(Evaluation) 함수 (수정됨)
# ----------------------------------------------------
def evaluate(model, data_loader, device='cpu'):
    """
    모델을 평가하고 Accuracy와 Error Correction Rate(ECR)를 계산합니다
    
    - Accuracy: 전체 큐비트 중 상태(에러 유무)를 맞춘 비율
    - ECR: 실제로 에러가 난 큐비트 중 에러라고 맞춘 비율 (Recall)
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    
    # Accuracy 계산 변수
    total_correct_qubits = 0
    total_qubits_count = 0
    
    # ECR 계산 변수
    total_error_qubits = 0      # 실제 에러가 난 큐비트 총 개수 (TP + FN)
    corrected_error_qubits = 0  # 그 중 맞춘 개수 (TP)

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Logit -> 0/1 예측 변환 (Threshold 0.0 == Sigmoid 0.5)
            preds = (outputs > 0).float()
            
            # 1. Overall Accuracy 계산
            # (preds == labels)는 [Batch, Num_Qubits] 크기의 True/False 행렬입니다
            total_correct_qubits += (preds == labels).sum().item()
            total_qubits_count += labels.numel() # 배치 크기 * 큐비트 수
            
            # 2. Error Correction Rate (ECR) 계산
            # 실제 에러가 있는 위치(Label=1)를 찾습니다
            error_mask = (labels == 1)
            
            total_error_qubits += error_mask.sum().item()
            
            # 실제 에러가 있는 곳에서 예측도 1인 경우를 셉니다
            corrected_error_qubits += (preds[error_mask] == 1).sum().item()

    avg_loss = total_loss / len(data_loader)
    
    # 분모가 0인 경우 방지
    accuracy = total_correct_qubits / total_qubits_count if total_qubits_count > 0 else 0.0
    ecr = corrected_error_qubits / total_error_qubits if total_error_qubits > 0 else 0.0

    return avg_loss, accuracy, ecr