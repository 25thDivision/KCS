import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, d_model=128, num_heads=4, num_layers=3):
        """
        Args:
            num_nodes (int): 입력 노드 개수 (45)
            in_channels (int): 노드 피처 차원 (6: Syndrome+RGB+Coord)
            num_qubits (int): 출력 큐비트 개수 (28)
            d_model (int): 트랜스포머 내부 차원 (기본 128)
            num_heads (int): 어텐션 헤드 개수 (기본 4)
            num_layers (int): 트랜스포머 레이어 깊이 (기본 3)
        """
        super(GraphTransformer, self).__init__()
        
        # 1. 임베딩 (Feature 6 -> d_model 128)
        self.embedding = nn.Linear(in_channels, d_model)
        
        # 2. 트랜스포머 인코더 (핵심)
        # batch_first=True: 입력 형태가 (Batch, Seq, Feature)임
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model*4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 출력 헤드 (Decoding)
        # 모든 노드의 정보를 일렬로 펼쳐서(Flatten) 판단
        self.flatten_dim = num_nodes * d_model
        
        self.output_head = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, num_qubits) # 최종 출력: 28개 큐비트의 Logits
        )

    def forward(self, x):
        """
        x: [Batch, Num_Nodes, In_Channels]
        (Graph Transformer는 엣지 정보(adj) 없이 노드 간 어텐션으로 학습합니다)
        """
        # (1) 임베딩
        x = self.embedding(x) # -> [Batch, 45, 128]
        
        # (2) 트랜스포머 인코딩 (Self-Attention)
        # 여기서 모든 노드가 서로의 정보를 참조합니다.
        x = self.transformer_encoder(x) # -> [Batch, 45, 128]
        
        # (3) Flatten (정보 압축)
        x = x.reshape(x.size(0), -1) # -> [Batch, 45 * 128]
        
        # (4) 예측
        logits = self.output_head(x) # -> [Batch, 28]
        
        return logits