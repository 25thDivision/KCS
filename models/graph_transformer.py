import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphTransformer(nn.Module):
    # [수정] dropout 인자 추가 (기본값 0.1)
    def __init__(self, num_nodes, in_channels, num_qubits, d_model=128, num_heads=4, num_layers=3, dropout=0.1):
        """
        Args:
            dropout (float): 드롭아웃 비율 (외부 설정값 반영)
        """
        super(GraphTransformer, self).__init__()
        
        self.embedding = nn.Linear(in_channels, d_model)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model*4, 
            batch_first=True,
            dropout=dropout # [수정] 입력받은 dropout 적용
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.flatten_dim = num_nodes * d_model
        
        self.output_head = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout), # [수정] 입력받은 dropout 적용
            nn.Linear(d_model * 2, num_qubits)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        logits = self.output_head(x)
        return logits