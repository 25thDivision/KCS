"""
IBM 신드롬 → Stim 호환 형태 변환기 (Surface Code용)

Data qubit 인덱스를 자동 추출하여, 모델 출력에서 올바른 위치만 가져올 수 있게 합니다.
Stim의 rotated surface code는 (홀수, 홀수) 좌표가 data qubit입니다.
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
    def __init__(self, distance: int, num_rounds: int, edge_dir: str = None):
        self.distance = distance
        self.num_rounds = num_rounds
        self.edge_dir = edge_dir

        self.stim_circuit = self._create_stim_reference()
        self.num_stim_detectors = self.stim_circuit.num_detectors
        self.num_stim_qubits = self.stim_circuit.num_qubits

        # Data qubit 인덱스 추출
        self.stim_data_qubit_indices = self._extract_data_qubit_indices()

        self.graph_mapper = None
        self.image_mapper = None
        self.edge_index = None

        self._initialize_mappers()

        print(f"[StimFormatConverter] Initialized for d={distance}, rounds={num_rounds}")
        print(f"    Stim total qubits: {self.num_stim_qubits}")
        print(f"    Stim data qubits: {len(self.stim_data_qubit_indices)} at indices {self.stim_data_qubit_indices}")
        print(f"    Stim detectors: {self.num_stim_detectors}")
        print(f"    Graph nodes: {self.graph_mapper.num_nodes}")

    def _create_stim_reference(self):
        try:
            from generators.surface_code import create_surface_code_circuit
        except ImportError:
            try:
                from simulation.generators.surface_code import create_surface_code_circuit
            except ImportError:
                import stim
                return stim.Circuit.generated(
                    "surface_code:rotated_memory_x",
                    distance=self.distance,
                    rounds=self.num_rounds,
                    before_round_data_depolarization=0.001
                )
        return create_surface_code_circuit(self.distance, self.num_rounds, 0.001)

    def _extract_data_qubit_indices(self) -> List[int]:
        """
        Stim 회로에서 data qubit 인덱스를 추출합니다.

        Rotated surface code에서 (홀수, 홀수) 좌표가 data qubit입니다.
        """
        data_indices = []

        for instruction in self.stim_circuit.flattened():
            if instruction.name == "QUBIT_COORDS":
                coords = instruction.gate_args_copy()
                targets = instruction.targets_copy()
                for t in targets:
                    x, y = coords[0], coords[1]
                    if int(x) % 2 == 1 and int(y) % 2 == 1:
                        data_indices.append(t.value)

        data_indices.sort()
        return data_indices

    def get_data_qubit_indices(self) -> List[int]:
        """모델 출력에서 data qubit에 해당하는 인덱스를 반환합니다."""
        return self.stim_data_qubit_indices

    def _initialize_mappers(self):
        from common.mapper_graph import SyndromeGraphMapper
        from common.mapper_image import SyndromeImageMapper

        self.graph_mapper = SyndromeGraphMapper(self.stim_circuit)
        self.image_mapper = SyndromeImageMapper(self.stim_circuit)
        self.edge_index = self.graph_mapper.get_edges()

        if self.edge_dir is not None:
            edge_path = os.path.join(self.edge_dir, f"edges_d{self.distance}.npy")
        else:
            edge_path = os.path.join(
                stim_dir, "dataset", "surface_code", "graph",
                f"edges_d{self.distance}.npy"
            )

        if os.path.exists(edge_path):
            self.edge_index = np.load(edge_path)
            print(f"    Loaded Phase 1 edges from: {edge_path}")
        else:
            print(f"    [Warning] Edge file not found: {edge_path}")
            print(f"    Using edges from SyndromeGraphMapper instead.")

    def ionq_to_stim_detectors(self, raw_syndromes: np.ndarray) -> np.ndarray:
        N, num_rounds, num_stab = raw_syndromes.shape

        detectors = np.zeros((N, num_rounds, num_stab), dtype=np.float32)
        detectors[:, 0, :] = raw_syndromes[:, 0, :]
        for r in range(1, num_rounds):
            detectors[:, r, :] = (raw_syndromes[:, r, :] != raw_syndromes[:, r - 1, :]).astype(np.float32)

        flat_detectors = detectors.reshape(N, -1)

        if flat_detectors.shape[1] < self.num_stim_detectors:
            padded = np.zeros((N, self.num_stim_detectors), dtype=np.float32)
            padded[:, :flat_detectors.shape[1]] = flat_detectors
            flat_detectors = padded
            print(f"    [Warning] Padded to {self.num_stim_detectors} detectors")
        elif flat_detectors.shape[1] > self.num_stim_detectors:
            flat_detectors = flat_detectors[:, :self.num_stim_detectors]
            print(f"    [Warning] Truncated to {self.num_stim_detectors} detectors")

        return flat_detectors

    def to_graph_format(self, raw_syndromes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        stim_detectors = self.ionq_to_stim_detectors(raw_syndromes)
        node_features = self.graph_mapper.map_to_node_features(stim_detectors)
        return node_features, self.edge_index

    def to_image_format(self, raw_syndromes: np.ndarray) -> np.ndarray:
        stim_detectors = self.ionq_to_stim_detectors(raw_syndromes)
        images = self.image_mapper.map_to_images(stim_detectors)
        return images

    def get_model_input_shape(self, model_type: str) -> tuple:
        if model_type == "graph":
            return (self.graph_mapper.num_nodes, 6)
        elif model_type == "image":
            return (1, self.image_mapper.height, self.image_mapper.width)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")