import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, hidden_dim, heads, num_layers, dropout):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_dim = hidden_dim // heads
        self.dropout = dropout
        
        self.lin_in = nn.Linear(in_channels, hidden_dim)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.hidden_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.hidden_dim))
        
        # [NEW] 출력 헤드
        self.output_head = nn.Linear(num_nodes * hidden_dim, num_qubits)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj):
        B, N, _ = x.shape
        x = self.lin_in(x)
        x = x.view(B, N, self.heads, self.hidden_dim)
        
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        scores = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)
        scores = self.leaky_relu(scores)
        
        mask = adj.unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        alpha = F.softmax(scores, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        out = torch.einsum("bijh,bjhd->bihd", alpha, x)
        out = out.reshape(B, N, -1) # (B, N, Hidden)
        
        # [수정] Flatten -> Output
        out = out.view(B, -1)
        out = self.output_head(out)
        return out