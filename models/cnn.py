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
        # 커널 크기 3x3, 스트라이드 1을 사용하여 지역적 특징을 추출합니다.
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0)
        
        # Pooling Layer (학습 파라미터는 없으므로 층 개수에는 보통 포함하지 않습니다)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # FC Layer 진입을 위한 차원 계산
        # (H, W) -> Conv -> (H-2, W-2) -> Pool -> ((H-2)/2, (W-2)/2)
        conv_height = height - 2
        conv_width = width - 2
        pool_height = (conv_height - 2) // 2 + 1
        pool_width = (conv_width - 2) // 2 + 1
        
        self.flattened_size = 16 * pool_height * pool_width
        
        # [Layer 2] Hidden Fully Connected Layer
        self.fc1 = nn.Linear(self.flattened_size, 64)
        
        # [Layer 3] Output Fully Connected Layer
        # 최종적으로 각 큐비트의 에러 확률을 출력합니다.
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # 1. Feature Extraction (Conv -> ReLU -> Pool)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 2. Flatten (1D 벡터로 펼치기)
        x = torch.flatten(x, 1) 
        
        # 3. Classification (FC -> ReLU -> FC)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        
        return x