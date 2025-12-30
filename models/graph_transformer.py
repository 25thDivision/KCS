import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, d_model=128, num_heads=4, num_layers=3, dropout=0.1):
        super(GraphTransformer, self).__init__()
        
        # 1. 입력 피처 임베딩
        self.embedding = nn.Linear(in_channels, d_model)
        
        # [핵심] Sinusoidal Positional Encoding (학습 X, 고정값 O)
        # 랜덤 초기화보다 훨씬 빠르게 수렴하며, 위치 정보를 명확히 줍니다.
        pe = torch.zeros(num_nodes, d_model)
        position = torch.arange(0, num_nodes, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 모델의 state_dict에 저장되지만 gradient 업데이트는 안 됨 (buffer)
        self.register_buffer('pe', pe.unsqueeze(0)) 
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model*4, 
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.flatten_dim = num_nodes * d_model
        
        self.output_head = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_qubits)
        )

    def forward(self, x):
        # x: [Batch, Num_Nodes, Features]
        x = self.embedding(x)
        
        # [핵심] 고정된 위치 정보 더하기 (Broadcasting)
        # x와 pe의 차원이 맞아야 하므로 num_nodes가 고정되어 있어야 합니다.
        x = x + self.pe[:, :x.size(1), :]
        
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        logits = self.output_head(x)
        return logits