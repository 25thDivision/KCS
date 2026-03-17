"""
Phase 1에서 학습된 ML 모델을 로드하여 IonQ 신드롬에 대해 추론합니다.

Phase 1의 config.json에서 모델 하이퍼파라미터를 읽어와
정확히 동일한 아키텍처로 모델을 재구성한 후, 학습된 가중치를 로드합니다.
"""

import os
import sys
import json
import torch
import numpy as np

# Phase 1 모델 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
ionq_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(ionq_dir)
stim_dir = os.path.join(root_dir, "stim_simulation")
sys.path.append(stim_dir)


def _load_phase1_model_config(model_name: str) -> dict:
    """Phase 1의 config.json에서 모델 하이퍼파라미터를 읽어옵니다."""
    config_path = os.path.join(stim_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Phase 1 config.json not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    models = config.get("models", config)  # "models" 키가 있으면 사용, 없으면 전체 사용
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in Phase 1 config.json")
    
    return models[model_name]


class MLDecoderAdapter:
    """
    Phase 1에서 학습된 ML 모델을 Phase 2 파이프라인에 연결합니다.
    
    Parameters:
        model_name (str): 모델 이름 (CNN, UNet, GraphMamba 등)
        weight_path (str): .pth 체크포인트 파일 경로
        model_type (str): "image" 또는 "graph"
        distance (int): Code distance
        input_shape (tuple): 모델 입력 shape (StimFormatConverter에서 제공)
            - graph: (num_nodes, feature_dim)
            - image: (1, H, W)
        num_qubits (int): 출력 차원 (데이터 큐빗 수). None이면 체크포인트에서 추론.
        device (str): "cuda" 또는 "cpu"
    """
    
    def __init__(self, model_name: str, weight_path: str,
                 model_type: str = "graph",
                 distance: int = 3,
                 input_shape: tuple = None,
                 num_qubits: int = None,
                 device: str = None):
        self.model_name = model_name
        self.weight_path = weight_path
        self.model_type = model_type
        self.distance = distance
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 체크포인트 로드
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Weight file not found: {self.weight_path}")
        
        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.state_dict = checkpoint["model_state_dict"]
        else:
            self.state_dict = checkpoint
        
        # 출력 차원 결정
        if num_qubits is None:
            num_qubits = self._infer_output_dim()
        self.num_qubits = num_qubits
        
        # 입력 shape 결정
        if input_shape is None:
            input_shape = self._infer_input_shape()
        self.input_shape = input_shape
        
        # Phase 1 config에서 하이퍼파라미터 로드
        self.phase1_config = _load_phase1_model_config(model_name)
        
        # 모델 생성 + 가중치 로드
        self.model = self._create_model()
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[MLDecoderAdapter] {model_name}: input={input_shape}, output={num_qubits}, device={self.device}")
    
    def _infer_output_dim(self) -> int:
        """State dict에서 출력 차원을 추론합니다."""
        for key in ["output_head.weight", "fc_out.weight", "final.weight"]:
            if key in self.state_dict:
                return self.state_dict[key].shape[0]
        last_weight = [k for k in self.state_dict.keys() if "weight" in k][-1]
        return self.state_dict[last_weight].shape[0]
    
    def _infer_input_shape(self) -> tuple:
        """State dict의 첫 번째 레이어에서 입력 shape을 추론합니다."""
        for key in sorted(self.state_dict.keys()):
            if "weight" in key and self.state_dict[key].dim() >= 2:
                in_features = self.state_dict[key].shape[-1]
                if self.model_type == "graph":
                    return (in_features, 6)  # 대략적 추정
                else:
                    return (1, 8, 8)  # 대략적 추정
        return (6, 6) if self.model_type == "graph" else (1, 8, 8)
    
    def _create_model(self):
        """Phase 1 config의 하이퍼파라미터로 모델을 생성합니다."""
        from models.cnn import CNN
        from models.unet import UNet
        from models.gcn import GCN
        from models.gcnii import GCNII
        from models.gat import GAT
        from models.appnp import APPNP
        from models.gnn import GNN
        from models.graph_transformer import GraphTransformer
        from models.graph_mamba import GraphMamba
        
        params = self.phase1_config["params"]
        
        if self.model_name == "CNN":
            return CNN(height=self.input_shape[1], width=self.input_shape[2],
                      in_channels=1, num_classes=self.num_qubits)
        
        elif self.model_name == "UNet":
            return UNet(in_ch=1, out_ch=self.num_qubits,
                       base_filters=params.get("base_filters", 32))
        
        elif self.model_name == "GCN":
            return GCN(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                      num_qubits=self.num_qubits,
                      hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
        
        elif self.model_name == "GCNII":
            return GCNII(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                        num_qubits=self.num_qubits,
                        hidden_dim=params["hidden_dim"], num_layers=params["num_layers"],
                        alpha=params["alpha"], theta=params["theta"],
                        dropout=params["dropout"])
        
        elif self.model_name == "GAT":
            return GAT(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                      num_qubits=self.num_qubits,
                      hidden_dim=params["hidden_dim"], heads=params["heads"],
                      num_layers=params["num_layers"], dropout=params["dropout"])
        
        elif self.model_name == "APPNP":
            return APPNP(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                        num_qubits=self.num_qubits,
                        hidden_dim=params["hidden_dim"], K=params["K"],
                        alpha=params["alpha"])
        
        elif self.model_name == "GNN":
            return GNN(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                      num_qubits=self.num_qubits,
                      hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
        
        elif self.model_name == "GraphTransformer":
            return GraphTransformer(
                num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                num_qubits=self.num_qubits,
                d_model=params["d_model"], num_heads=params["num_heads"],
                num_layers=params["num_layers"], dropout=params["dropout"])
        
        elif self.model_name == "GraphMamba":
            return GraphMamba(
                num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                num_qubits=self.num_qubits,
                d_model=params["d_model"], num_layers=params["num_layers"],
                dropout=params["dropout"])
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def decode(self, syndromes: np.ndarray, edge_index: np.ndarray = None) -> np.ndarray:
        """
        신드롬 → 에러 위치 추정
        
        Args:
            syndromes: 전처리된 신드롬 배열
                - graph: (N, num_nodes, feature_dim)
                - image: (N, 1, H, W)
            edge_index: Graph 모델용 엣지 인덱스 (2, num_edges)
        
        Returns:
            corrections: (N, num_qubits) 에러 추정 벡터 (0 또는 1)
        """
        with torch.no_grad():
            inputs = torch.FloatTensor(syndromes).to(self.device)
            
            if self.model_type == "graph" and edge_index is not None:
                edge_idx = torch.LongTensor(edge_index).to(self.device)
                outputs = self.model(inputs, edge_idx)
            else:
                outputs = self.model(inputs)
            
            predictions = (outputs > 0).float().cpu().numpy()
        
        return predictions.astype(np.int8)
