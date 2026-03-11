import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # 1. 메시지 전달 (이웃 노드 정보 합치기)
        out = torch.matmul(adj, x)
        # 2. 가중치 적용
        out = self.linear(out)
        return out

class GNN(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, hidden_dim=64, num_layers=2):
        """
        Args:
            num_nodes: 노드 개수
            in_channels: 입력 피처 차원
            num_qubits: 출력 큐비트 개수
            hidden_dim: 은닉층 차원
            num_layers (int): GNN 레이어 깊이 (Depth) <-- 추가됨
        """
        super(GNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # GNN 레이어들을 리스트에 담습니다 (ModuleList)
        self.layers = nn.ModuleList()
        
        # 첫 번째 레이어 (Input -> Hidden)
        self.layers.append(GNNLayer(in_channels, hidden_dim))
        
        # 나머지 레이어들 (Hidden -> Hidden)
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
        
        # 출력층
        self.fc = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_qubits)
        )

    def forward(self, x, edge_index):
        device = x.device
        batch_size = x.size(0)
        
        # 1. 인접 행렬(Adjacency Matrix) 생성
        adj = torch.eye(self.num_nodes, device=device) # Self-loop
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
        
        # 정규화
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj = adj / degree 

        # 2. GNN 레이어 통과 (반복문)
        for layer in self.layers:
            x = F.relu(layer(x, adj))
        
        # 3. Flatten
        x = x.reshape(batch_size, -1)
        
        # 4. 예측
        logits = self.fc(x)
        return logits