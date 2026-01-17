import torch
import torch.nn as nn
import torch.nn.functional as F

class APPNP(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, hidden_dim, K, alpha):
        super(APPNP, self).__init__()
        self.K = K
        self.alpha = alpha
        
        self.lin1 = nn.Linear(in_channels, hidden_dim)
        # [NEW] 출력 헤드
        self.output_head = nn.Linear(num_nodes * hidden_dim, num_qubits)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        # 1. Feature Transform
        x = self.lin1(x)
        x = self.activation(x)
        
        # 2. Propagation
        h = x
        for _ in range(self.K):
            aggregated = torch.matmul(adj, h)
            h = (1 - self.alpha) * aggregated + self.alpha * x
            
        # [수정] Flatten -> Output
        h = h.view(h.size(0), -1)
        h = self.output_head(h)
        return h