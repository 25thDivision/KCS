import torch
from torch.utils.data import Dataset
import numpy as np

class QECDataset(Dataset):
    """
    Numpy 형태의 신드롬 데이터와 에러 라벨을 PyTorch 텐서로 변환하여 제공하는 데이터셋 클래스입니다
    """
    def __init__(self, features, labels):
        # ✅ [수정] 입력이 이미 PyTorch 텐서(GPU 등)라면 변환 없이 그대로 씁니다.
        if torch.is_tensor(features):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        if torch.is_tensor(labels):
            self.labels = labels
        else:
            self.labels = torch.FloatTensor(labels) 
            # 만약 labels가 정수형이어야 한다면 torch.LongTensor(labels) 사용

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]