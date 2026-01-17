import torch
import torch.nn as nn

# CUDA 사용 가능 여부 확인
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

try:
    # mamba_ssm 로드 시도
    from mamba_ssm import Mamba
    print("Mamba-SSM Import Success!")

    # 간단한 Mamba 레이어 생성 (GPU로 이동)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # (batch, seq_len, d_model)
    x = torch.randn(2, 64, 16).to(device)
    
    # Mamba Layer 정의
    mamba_layer = Mamba(
        d_model=16, 
        d_state=16, 
        d_conv=4, 
        expand=2
    ).to(device)
    
    # Forward Pass 실행
    y = mamba_layer(x)
    print("Forward Pass Success!")
    print(f"Output Shape: {y.shape}") # 예상: (2, 64, 16)
    
except Exception as e:
    print(f"Error Occurred: {e}")