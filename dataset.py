import torch
from torch.utils.data import Dataset
import numpy as np

class QECDataset(Dataset):
    """
    Numpy 형태의 신드롬 데이터와 에러 라벨을 PyTorch 텐서로 변환하여 제공하는 데이터셋 클래스입니다
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, 1, H, W) for CNN or (N, Nodes, Features) for GNN
            labels: (N, Num_Qubits) 물리적 에러 라벨
        """
        # CNN용 이미지는 float32, 라벨도 BCE Loss 계산을 위해 float32로 변환합니다
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]