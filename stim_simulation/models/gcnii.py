import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNII(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, hidden_dim, num_layers, alpha, theta, dropout):
        super(GCNII, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        self.dropout = dropout
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_dim))
        for _ in range(num_layers):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            
        # [NEW] 출력 헤드
        self.output_head = nn.Linear(num_nodes * hidden_dim, num_qubits)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        x = self.lins[0](x)
        x = self.activation(x)
        x_0 = x 

        for i in range(self.num_layers):
            support = torch.matmul(adj, x)
            initial = (1 - self.alpha) * support + self.alpha * x_0
            tmp = self.lins[i+1](initial)
            x = (1 - self.theta) * initial + self.theta * tmp
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # [수정] Flatten -> Output
        x = x.view(x.size(0), -1)
        x = self.output_head(x)
        return x