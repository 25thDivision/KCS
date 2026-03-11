import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphTransformerLayer(nn.Module):
    """
    그래프 구조를 인식하는 Transformer Layer입니다.
    기본 TransformerEncoderLayer와 달리, Adjacency Mask를 사용하여
    연결된 이웃 노드 간에만 Attention을 수행합니다.
    """
    def __init__(self, d_model, num_heads, dropout=0.1, dim_feedforward=None):
        super(GraphTransformerLayer, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        src: (Batch, Num_Nodes, Hidden_Dim)
        src_mask: (Batch*Num_Heads, Num_Nodes, Num_Nodes) 또는 (Num_Nodes, Num_Nodes)
                  연결되지 않은 부분은 -inf, 연결된 부분은 0인 마스크
        """
        # 1. Multi-Head Attention with Graph Mask
        # attn_output: (Batch, Num_Nodes, Hidden_Dim)
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 2. Feed Forward Network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, in_channels, num_qubits, d_model=128, num_heads=4, num_layers=3, dropout=0.1):
        """
        Args:
            num_nodes: 그래프 노드 개수
            in_channels: 입력 피처 차원 (예: 6)
            num_qubits: 출력 큐비트 개수 (예: 물리적 에러 위치)
            d_model: 히든 차원
            num_heads: 어텐션 헤드 개수
            num_layers: 레이어 깊이
            dropout: 드롭아웃 비율
        """
        super(GraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # 1. Embedding Layer
        self.embedding = nn.Linear(in_channels, d_model)
        
        # 2. Graph Transformer Layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # 3. Output Head
        self.output_head = nn.Sequential(
            nn.Linear(num_nodes * d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_qubits)
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: Node Features (Batch, Num_Nodes, In_Channels)
            edge_index: Graph Edges (2, Num_Edges) - from PyG style or loaded from .npy
        """
        batch_size = x.size(0)
        device = x.device
        
        # 1. Adjacency Mask 생성 (Attention Mask용)
        # 연결된 곳은 0, 연결되지 않은 곳은 -inf
        # 매번 생성하면 느리므로 캐싱하거나, GNN 방식의 Message Passing으로 바꿀 수도 있지만
        # 현재 구조(Transformer)를 유지하며 가장 정확한 방법입니다.
        
        # (Num_Nodes, Num_Nodes) 크기의 마스크 생성
        # 초기값: -inf (모두 차단)
        attn_mask = torch.full((self.num_nodes, self.num_nodes), float('-inf'), device=device)
        
        # 자기 자신(Self-loop)은 항상 연결
        attn_mask.fill_diagonal_(0.0)
        
        # Edge 정보로 연결된 부분 0.0으로 설정
        # edge_index가 (2, E) 형태라고 가정
        if edge_index is not None and edge_index.shape[1] > 0:
            src, dst = edge_index
            attn_mask[src, dst] = 0.0
            attn_mask[dst, src] = 0.0 # 무방향 그래프 가정
            
        # 2. Forward Passing
        x = self.embedding(x) # (B, N, C) -> (B, N, d_model)
        
        for layer in self.layers:
            # Mask를 넣어주어 이웃끼리만 Attention하게 만듦
            x = layer(x, src_mask=attn_mask)
            
        # 3. Flatten & Output
        x = x.reshape(batch_size, -1)
        logits = self.output_head(x)
        
        return logits