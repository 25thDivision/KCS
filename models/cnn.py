# models/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, height, width, in_channels=1, num_observables=1):
        """
        'd'값에 상관없이 범용적으로 작동하는 CNN 모델
        
        Args:
            height (int): 입력 이미지의 높이 (e.g., d=5일 때 6)
            width (int): 입력 이미지의 너비 (e.g., d=5일 때 5)
        """
        super(SimpleCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_observables = num_observables
        
        # 1. Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 2. FC Layer를 만들기 위해, Conv/Pool을 거친 후의
        #    최종 H'과 W' 크기를 동적으로 계산합니다.
        
        # (N, C, H, W) -> (N, 16, H-2, W-2)
        conv_height = height - 2
        conv_width = width - 2
        
        # (N, 16, H-2, W-2) -> (N, 16, H'/2, W'/2)
        pool_height = (conv_height - 2) // 2 + 1
        pool_width = (conv_width - 2) // 2 + 1
        
        # 최종 flattened_size 계산
        self.flattened_size = 16 * pool_height * pool_width
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc_out = nn.Linear(64, num_observables)

    def forward(self, x):
        """
        x: (Batch, 1, H, W)
        """
        # Conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Flatten
        x = torch.flatten(x, 1) # (Batch, 16 * H' * W')
        
        # FC Layers
        x = self.fc1(x)
        x = F.relu(x)
        
        # Output Layer
        x = self.fc_out(x)
        
        return x