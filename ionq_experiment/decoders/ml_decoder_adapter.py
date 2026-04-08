"""
Phase 1에서 학습된 ML 모델을 로드하여 신드롬에 대해 추론합니다.

use_adj 모델(GCN, GCNII, GAT, APPNP)은 edge_index를 adj matrix로 변환하여 사용합니다.
"""

import os
import sys
import json
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(experiment_dir)
stim_dir = os.path.join(root_dir, "stim_simulation")
sys.path.append(stim_dir)
sys.path.append(root_dir)

from paths import ProjectPaths

PATHS = ProjectPaths(root_dir)


def _load_phase1_model_config(model_name: str) -> dict:
    config = PATHS.load_stim_config()
    models = config.get("models", config)
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in Phase 1 config.json")
    return models[model_name]


class MLDecoderAdapter:
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

        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Weight file not found: {self.weight_path}")

        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.state_dict = checkpoint["model_state_dict"]
        else:
            self.state_dict = checkpoint

        if num_qubits is None:
            num_qubits = self._infer_output_dim()
        self.num_qubits = num_qubits

        if input_shape is None:
            input_shape = self._infer_input_shape()
        self.input_shape = input_shape

        self.phase1_config = _load_phase1_model_config(model_name)
        self.use_adj = self.phase1_config.get("use_adj", False)

        self.model = self._create_model()
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"[MLDecoderAdapter] {model_name}: input={input_shape}, output={num_qubits}, "
              f"use_adj={self.use_adj}, device={self.device}")

    def _infer_output_dim(self) -> int:
        for key in ["output_head.weight", "fc_out.weight", "final.weight"]:
            if key in self.state_dict:
                return self.state_dict[key].shape[0]
        last_weight = [k for k in self.state_dict.keys() if "weight" in k][-1]
        return self.state_dict[last_weight].shape[0]

    def _infer_input_shape(self) -> tuple:
        for key in sorted(self.state_dict.keys()):
            if "weight" in key and self.state_dict[key].dim() >= 2:
                in_features = self.state_dict[key].shape[-1]
                if self.model_type == "graph":
                    return (in_features, 6)
                else:
                    return (1, 8, 8)
        return (6, 6) if self.model_type == "graph" else (1, 8, 8)

    def _create_model(self):
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
                       in_channels=self.input_shape[0], num_classes=self.num_qubits)
        elif self.model_name == "UNet":
            return UNet(in_ch=self.input_shape[0], out_ch=self.num_qubits,
                        base_filters=params.get("base_filters", 32))
        elif self.model_name == "GCN":
            return GCN(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                       num_qubits=self.num_qubits,
                       hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
        elif self.model_name == "GCNII":
            return GCNII(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                         num_qubits=self.num_qubits,
                         hidden_dim=params["hidden_dim"], num_layers=params["num_layers"],
                         alpha=params["alpha"], theta=params["theta"], dropout=params["dropout"])
        elif self.model_name == "GAT":
            return GAT(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                       num_qubits=self.num_qubits,
                       hidden_dim=params["hidden_dim"], heads=params["heads"],
                       num_layers=params["num_layers"], dropout=params["dropout"])
        elif self.model_name == "APPNP":
            return APPNP(num_nodes=self.input_shape[0], in_channels=self.input_shape[1],
                         num_qubits=self.num_qubits,
                         hidden_dim=params["hidden_dim"], K=params["K"], alpha=params["alpha"])
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

    def _edge_index_to_adj(self, edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
        """edge_index (2, num_edges) → adj matrix (num_nodes, num_nodes)"""
        from torch_geometric.utils import to_dense_adj

        edge_idx = torch.LongTensor(edge_index).to(self.device)
        adj = to_dense_adj(edge_idx, max_num_nodes=num_nodes)[0]
        adj = adj + torch.eye(num_nodes, device=self.device)
        adj = (adj > 0).float()
        return adj

    def decode(self, syndromes: np.ndarray, edge_index: np.ndarray = None) -> np.ndarray:
        """
        신드롬 → 에러 위치 추정

        use_adj 모델은 edge_index를 adj matrix로 변환하여 사용합니다.
        """
        with torch.no_grad():
            inputs = torch.FloatTensor(syndromes).to(self.device)

            if self.model_type == "graph" and edge_index is not None:
                if self.use_adj:
                    num_nodes = inputs.shape[1]
                    adj = self._edge_index_to_adj(edge_index, num_nodes)
                    outputs = self.model(inputs, adj)
                else:
                    edge_idx = torch.LongTensor(edge_index).to(self.device)
                    outputs = self.model(inputs, edge_idx)
            else:
                outputs = self.model(inputs)

            predictions = (outputs > 0).float().cpu().numpy()

        return predictions.astype(np.int8)