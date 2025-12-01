import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, height, width, in_channels=1, num_classes=1):
        """
        Args:
            height (int): 입력 이미지의 높이 (d=5일 때 6)
            width (int): 입력 이미지의 너비 (d=5일 때 5)
            in_channels (int): 입력 채널 수 (보통 1)
            num_classes (int): 예측할 큐비트의 총 개수 (물리적 에러 위치 추적용)
        """
        super(CNN, self).__init__()
        
        # [Layer 1] Convolutional Layer
        # 커널 크기 3x3, 스트라이드 1, 패딩 1 (이미지 크기 유지)
        # 작은 이미지(6x5)에서 정보를 잃지 않으려면 padding=1을 주는 것이 좋습니다.
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        
        # [삭제됨] Pooling Layer
        # 이미지가 너무 작아서(6x5) 풀링을 하면 위치 정보가 사라지므로 제거했습니다.
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # FC Layer 진입을 위한 차원 계산 (Pooling 없음, Padding=1로 크기 유지)
        # (H, W) -> Conv(pad=1) -> (H, W) 그대로 유지됨
        conv_height = height
        conv_width = width
        
        self.flattened_size = 32 * conv_height * conv_width
        
        # [Layer 2] Hidden Fully Connected Layer
        self.fc1 = nn.Linear(self.flattened_size, 128)
        
        # [Layer 3] Output Fully Connected Layer
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. Feature Extraction (Conv -> ReLU)
        # 풀링(pool1) 과정이 빠졌습니다!
        x = self.conv1(x)
        x = F.relu(x)
        
        # 2. Flatten (1D 벡터로 펼치기)
        x = torch.flatten(x, 1) 
        
        # 3. Classification (FC -> ReLU -> FC)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        
        return x