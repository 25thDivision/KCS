import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, hidden_dim, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        # 첫 레이어
        self.layers.append(nn.Linear(in_channels, hidden_dim))
        # 히든 레이어
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # [수정] 마지막 GCN 레이어 (출력용 아님, 히든용)
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # [NEW] 출력 헤드: (N * Hidden) -> Num_Qubits (차원 불일치 해결사)
        self.output_head = nn.Linear(num_nodes * hidden_dim, num_qubits)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        # x: (Batch, Nodes, Features)
        
        # GCN 연산들
        x = self.layers[0](x)
        x = self.activation(x)
        x = torch.matmul(adj, x) 

        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
            x = torch.matmul(adj, x)
        
        # [수정] Flatten -> Linear Projection
        # (Batch, Nodes, Hidden) -> (Batch, Nodes * Hidden)
        x = x.view(x.size(0), -1)
        
        # (Batch, N*H) -> (Batch, Num_Qubits)
        x = self.output_head(x)
        return x