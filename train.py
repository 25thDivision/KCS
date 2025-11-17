import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # 학습 진행률을 보여주는 라이브러리

# ----------------------------------------------------
# 1. 손실 함수 (Loss Function) 정의
# ----------------------------------------------------
# Sigmoid를 거치지 않은 Logit 값을 입력받아 BCE Loss를 계산 (수치적으로 안정적)
loss_fn = nn.BCEWithLogitsLoss()


# ----------------------------------------------------
# 2. 1 에포크(Epoch) 학습 함수
# ----------------------------------------------------
def train_one_epoch(model, data_loader, optimizer):
    """
    모델을 1 에포크 동안 학습시킵니다.
    
    Args:
        model (nn.Module): 학습시킬 모델 (e.g., SimpleCNN_d5)
        data_loader (DataLoader): 학습용 데이터 로더 (배치 단위로 데이터 공급)
        optimizer (torch.optim.Optimizer): 옵티마이저 (e.g., Adam)
    
    Returns:
        float: 1 에포크 동안의 평균 Loss
    """
    model.train() # 모델을 "학습 모드"로 설정
    
    total_loss = 0.0
    
    # tqdm: 데이터 로더를 순회하며 진행 바(progress bar)를 표시
    for inputs, labels in tqdm(data_loader, desc="Training Epoch"):
        
        # 1. 옵티마이저의 그래디언트 초기화 (필수)
        optimizer.zero_grad()
        
        # 2. 순전파 (Forward Pass)
        # inputs는 (Batch, 1, 6, 5) 크기의 이미지 텐서
        outputs = model(inputs)
        
        # 3. 손실(Loss) 계산
        # labels는 (Batch, 1) 크기, outputs도 (Batch, 1) 크기여야 함
        loss = loss_fn(outputs, labels.float()) # Y가 float 타입이어야 함
        
        # 4. 역전파 (Backward Pass) - PyTorch가 알아서 그래디언트 계산
        loss.backward()
        
        # 5. 가중치 업데이트 (Weight Update) - Adam 로직이 여기서 자동으로 실행됨
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)


# ----------------------------------------------------
# 3. 평가(Evaluation) 함수 (Accuracy, ECR 측정용)
# ----------------------------------------------------
def evaluate(model, data_loader):
    """
    모델을 평가 데이터셋으로 평가합니다.
    
    Args:
        model (nn.Module): 평가할 모델
        data_loader (DataLoader): 평가용 데이터 로더
    
    Returns:
        tuple: (평균 Loss, 전체 Accuracy, Error Correction Rate)
    """
    model.eval() # 모델을 "평가 모드"로 설정 (Dropout 등 비활성화)
    
    total_loss = 0.0
    
    correct_predictions = 0
    total_samples = 0
    
    # ECR (Error Correction Rate) 계산을 위한 변수
    # (참고: ECR의 정확한 정의에 따라 로직이 복잡해질 수 있습니다.
    #       여기서는 "실제 논리 오류가 있었던(Label=1) 샘플" 중에서
    #       "정확히 맞춘(Prediction=1) 비율"로 가정합니다.)
    error_samples = 0
    correctly_corrected_errors = 0

    # torch.no_grad(): 그래디언트 계산을 멈춰 메모리/속도 최적화
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            
            # 1. 순전파 (Forward Pass)
            outputs = model(inputs) # Logit 값 (e.g., -2.5, 1.8, ...)
            
            # 2. 손실 계산
            loss = loss_fn(outputs, labels.float())
            total_loss += loss.item()
            
            # 3. Accuracy 계산
            # Logit을 0/1 예측으로 변환 (Sigmoid > 0.5 == Logit > 0)
            preds = (outputs > 0).float()
            
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # 4. ECR 계산
            # 실제 오류가 있었던(Label=1) 샘플들의 인덱스
            error_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            if error_indices.numel() > 0:
                error_samples += error_indices.numel()
                # 그 중에서 예측도 1로 맞춘 경우
                correctly_corrected_errors += (preds[error_indices] == 1).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    # ECR 계산 (0으로 나누는 오류 방지)
    ecr = (correctly_corrected_errors / error_samples) if error_samples > 0 else 0.0

    return avg_loss, accuracy, ecr