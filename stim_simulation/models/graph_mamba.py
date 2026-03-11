import torch
import torch.nn as nn
from torch_geometric.utils import degree

# Mamba 라이브러리 임포트 (설치 안 되어 있을 시 에러 메시지)
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class GraphMamba(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, d_model=64, num_layers=4, dropout=0.1):
        super(GraphMamba, self).__init__()
        
        if Mamba is None:
            raise ImportError("❌ mamba-ssm 라이브러리가 설치되지 않았습니다. `pip install mamba-ssm`을 실행하세요.")

        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # 1. Embedding
        self.embedding = nn.Linear(in_channels, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # 2. Mamba Layers (The core SSM)
        # 논문 구현: Mamba 블록을 쌓아 장거리 의존성 학습
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension
                d_state=16,      # SSM state expansion factor
                d_conv=4,        # Local convolution width
                expand=2,        # Block expansion factor
            ) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # 3. Prediction Head
        self.output_head = nn.Sequential(
            nn.Linear(num_nodes * d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_qubits)
        )

    def forward(self, x, edge_index):
        """
        x: (Batch, Num_Nodes, Features)
        edge_index: (2, Num_Edges) - 연결 정보
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device

        # --- [Step 1] Graph to Sequence (Node Prioritization) ---
        # 논문 아이디어: Degree가 높은 노드(허브)를 시퀀스 앞쪽(혹은 뒤쪽)에 배치하여 정보 흐름 최적화
        
        # 1. 노드별 차수(Degree) 계산
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        
        # 2. 차수가 높은 순서대로 정렬 (Descending sort)
        # 중요도가 높은 노드를 먼저 처리하도록 배치
        _, sort_idx = torch.sort(deg, descending=True)
        
        # 3. 나중에 원래 순서로 복구하기 위한 인덱스(Unsort Index) 생성
        unsort_idx = torch.argsort(sort_idx)

        # 4. 입력 데이터 정렬 (Batch, Sorted_Nodes, Features)
        x = x[:, sort_idx, :]
        
        # --- [Step 2] Mamba Encoding ---
        x = self.embedding(x)
        x = self.norm_in(x)

        for layer, norm in zip(self.layers, self.norms):
            # Mamba Forward
            x_out = layer(x)
            # Residual Connection + Norm + Dropout
            x = norm(x + self.dropout(x_out))

        # --- [Step 3] Sequence to Graph (Restoration) ---
        # 섞었던 순서를 원래 큐비트/신드롬 순서대로 복구해야 라벨(y)과 위치가 맞음
        x = x[:, unsort_idx, :]

        # --- [Step 4] Prediction ---
        x = x.reshape(batch_size, -1) # Flatten
        logits = self.output_head(x)
        
        return logits