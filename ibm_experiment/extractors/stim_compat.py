"""
IBM Heavy-Hex Surface Code 신드롬 → Stim 호환 형태 변환기

heavyhex_surface_code_depth7 회로에서 나온 HW syndrome을
Phase 1 ML 모델이 기대하는 입력 형태로 변환합니다.

핵심 변환:
  1. Temporal differencing (no-reset XOR tracking)
  2. HW syndrome 순서 → Stim TRUE stabilizer 순서 재배열
     (Z stab만 매핑, X stab은 0 마스킹 → reorder_hw_to_stim())
  3. SurfaceCodeGraphMapper / SurfaceCodeImageMapper로 ML 입력 생성

HW per-cycle 8 bits:
  [Z1, X1, Xb_right, Zb_top, Z2, X2, Xb_left, Zb_bot]

Stim per-round 8 bits:
  [Z{0134}, Z{1245}, Z{01}, Z{78}, X{0126}, X{2346}, X{25}, X{78}]
"""

import os
import sys
import numpy as np
from typing import Tuple, List

current_dir = os.path.dirname(os.path.abspath(__file__))
ibm_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(ibm_dir)
stim_dir = os.path.join(root_dir, "stim_simulation")
stim_sim_dir = os.path.join(stim_dir, "simulation")

sys.path.append(stim_dir)
sys.path.append(stim_sim_dir)


class StimFormatConverter:
    """
    IBM Heavy-Hex 신드롬을 Phase 1 ML 모델 입력으로 변환합니다.

    Parameters:
        distance: Code distance (3 only)
        num_rounds: QEC 라운드 수 (= num_cycles)
        edge_dir: Phase 1 edge 파일 디렉토리 (stim_data_dir)
        code_type: "heavyhex_surface_code" (기본값)
    """

    def __init__(self, distance: int, num_rounds: int,
                 edge_dir: str = None, code_type: str = "heavyhex_surface_code"):
        self.distance = distance
        self.num_rounds = num_rounds
        self.edge_dir = edge_dir
        self.code_type = code_type

        # Heavyhex 설정 로드
        from generators.heavyhex_surface_code import (
            HEAVYHEX_D3, create_heavyhex_surface_code_circuit
        )
        self.cfg = HEAVYHEX_D3
        self.num_stabilizers = self.cfg["num_ancilla"]  # 8

        # Stim 참조 회로 (detector 수 확인용)
        self.stim_circuit = create_heavyhex_surface_code_circuit(
            distance=distance, rounds=num_rounds, noise=0.001
        )
        self.num_stim_detectors = self.num_rounds * self.num_stabilizers

        # Data qubit indices (Stim 기준: 0-8)
        self.stim_data_qubit_indices = list(range(self.cfg["num_data"]))

        # Mapper 초기화
        self.graph_mapper = None
        self.image_mapper = None
        self.edge_index = None
        self._initialize_mappers()

        print(f"[StimFormatConverter] Initialized for heavyhex d={distance}, rounds={num_rounds}")
        print(f"    Stim detectors: {self.num_stim_detectors}")
        print(f"    Data qubits: {len(self.stim_data_qubit_indices)}")
        print(f"    Graph nodes: {self.graph_mapper.num_nodes}")

    def _initialize_mappers(self):
        """Heavyhex TRUE stabilizer 기반 mapper 초기화."""
        from common.mapper_surface import SurfaceCodeGraphMapper, SurfaceCodeImageMapper

        cfg = self.cfg
        z_stabs = [cfg["z_stabilizers"][a] for a in cfg["z_ancilla"]]
        x_stabs = [cfg["x_stabilizers"][a] for a in cfg["x_ancilla"]]

        self.graph_mapper = SurfaceCodeGraphMapper(
            self.distance, self.num_rounds, z_stabs, x_stabs
        )
        self.image_mapper = SurfaceCodeImageMapper(
            self.distance, self.num_rounds, self.num_stabilizers
        )
        self.edge_index = self.graph_mapper.get_edges()

        # Phase 1에서 저장한 edge 파일이 있으면 로드
        edge_path = None
        if self.edge_dir is not None:
            edge_path = os.path.join(self.edge_dir, f"edges_d{self.distance}.npy")
        else:
            edge_path = os.path.join(
                stim_dir, "dataset", self.code_type, "graph",
                f"edges_d{self.distance}.npy"
            )

        if edge_path and os.path.exists(edge_path):
            self.edge_index = np.load(edge_path)
            print(f"    Loaded Phase 1 edges: {edge_path}")
        else:
            print(f"    Using edges from SurfaceCodeGraphMapper")

    def get_data_qubit_indices(self) -> List[int]:
        """모델 출력에서 data qubit에 해당하는 인덱스."""
        return self.stim_data_qubit_indices

    def hw_to_stim_detectors(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        HW raw syndrome → Stim-compatible flat detector 배열로 변환.

        처리 순서:
          1. Temporal differencing (no-reset XOR tracking)
          2. HW → Stim 순서 재배열 (Z만 매핑, X 마스킹)

        Args:
            raw_syndromes: (N, num_rounds, 8) — HW 측정 순서 per-cycle

        Returns:
            stim_detectors: (N, num_rounds * 8) — Stim 순서, temporal-differenced
        """
        from generators.heavyhex_surface_code import reorder_hw_to_stim

        N, num_rounds, num_stab = raw_syndromes.shape

        # Step 1: Temporal differencing (no-reset 보정)
        detectors = np.zeros_like(raw_syndromes)
        detectors[:, 0, :] = raw_syndromes[:, 0, :]
        for r in range(1, num_rounds):
            detectors[:, r, :] = (
                raw_syndromes[:, r, :] != raw_syndromes[:, r - 1, :]
            ).astype(np.float32)

        # Step 2: Flatten → (N, num_rounds * 8)
        flat_hw = detectors.reshape(N, -1)

        # Step 3: HW → Stim 순서 재배열 (Z만 매핑, X = 0)
        flat_stim = reorder_hw_to_stim(flat_hw, num_rounds)

        return flat_stim

    def to_graph_format(self, raw_syndromes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        HW syndrome → Graph 모델 입력.

        Returns:
            node_features: (N, num_nodes, feature_dim)
            edge_index: (2, num_edges)
        """
        stim_detectors = self.hw_to_stim_detectors(raw_syndromes)
        node_features = self.graph_mapper.map_to_node_features(stim_detectors)
        return node_features, self.edge_index

    def to_image_format(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        HW syndrome → Image 모델 입력.

        Returns:
            images: (N, 1, H, W)
        """
        stim_detectors = self.hw_to_stim_detectors(raw_syndromes)
        images = self.image_mapper.map_to_images(stim_detectors)
        return images

    def get_model_input_shape(self, model_type: str) -> tuple:
        """모델 타입에 따른 예상 입력 shape."""
        if model_type == "graph":
            return (self.graph_mapper.num_nodes, 6)
        elif model_type == "image":
            return (self.num_rounds, self.image_mapper.height, self.image_mapper.width)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
