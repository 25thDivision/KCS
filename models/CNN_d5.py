import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_d5(nn.Module):
    def __init__(self, num_observables=1):
        """
        d=5 (6x5 입력)에 최적화된 CNN 모델
        """
        super(SimpleCNN_d5, self).__init__()
        
        # 1. Convolutional Layer (특징 추출)
        # (Batch, 1, 6, 5) -> (Batch, 16, 4, 3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        
        # (Batch, 16, 4, 3) -> (Batch, 16, 2, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 2. Fully Connected Layers (분류)
        # 최종 특징 맵(16, 2, 1)을 1D로 펼친 크기
        # flattened_size = 16 * 2 * 1 = 32
        self.flattened_size = 32
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc_out = nn.Linear(64, num_observables)

    def forward(self, x):
        """
        x: (Batch, 1, 6, 5)
        """
        # Conv 1
        x = self.conv1(x)  # -> (Batch, 16, 4, 3)
        x = F.relu(x)
        x = self.pool1(x)  # -> (Batch, 16, 2, 1)
        
        # Flatten
        x = torch.flatten(x, 1) # -> (Batch, 32)
        
        # FC Layers
        x = self.fc1(x)
        x = F.relu(x)
        
        # Output Layer
        x = self.fc_out(x)
        
        return x