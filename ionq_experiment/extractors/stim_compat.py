"""
IonQ 신드롬 → Stim 호환 형태 변환기

Phase 1의 Stim 파이프라인에서 사용한 SyndromeGraphMapper / SyndromeImageMapper를
그대로 재활용하여, IonQ 측정 결과를 Phase 1 모델이 기대하는 입력 형태로 변환합니다.

변환 과정:
  1. IonQ 원시 앵실러 측정값 (per-round) 을 Stim-style 디텍터로 변환
     - Round 0: detector = raw measurement (초기 상태는 0이므로)
     - Round r: detector = raw[r] XOR raw[r-1] (연속 라운드 차분)
  2. 1D 디텍터 배열을 Stim의 GraphMapper / ImageMapper에 입력
  3. 모델이 기대하는 (N, num_nodes, feature_dim) 또는 (N, 1, H, W) 형태 출력
"""

import os
import sys
import numpy as np
from typing import Tuple

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
ionq_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(ionq_dir)
stim_dir = os.path.join(root_dir, "stim_simulation")
stim_sim_dir = os.path.join(stim_dir, "simulation")

sys.path.append(stim_dir)
sys.path.append(stim_sim_dir)


class StimFormatConverter:
    """
    IonQ 신드롬을 Phase 1 모델 입력 형태로 변환합니다.

    Parameters:
        distance (int): Code distance
        num_rounds (int): QEC 라운드 수 (Phase 1 학습 시와 동일해야 함)
        edge_dir (str): Phase 1 엣지 파일이 있는 디렉토리 경로
                        호출 시 PATHS.stim_data_dir(code_type, measurement, "graph") 전달
    """

    def __init__(self, distance: int, num_rounds: int, edge_dir: str = None):
        self.distance = distance
        self.num_rounds = num_rounds
        self.edge_dir = edge_dir

        # Stim 참조 회로 생성 (Phase 1과 동일한 파라미터)
        self.stim_circuit = self._create_stim_reference()
        self.num_stim_detectors = self.stim_circuit.num_detectors

        # Stim 매퍼 초기화
        self.graph_mapper = None
        self.image_mapper = None
        self.edge_index = None

        self._initialize_mappers()

        print(f"[StimFormatConverter] Initialized for d={distance}, rounds={num_rounds}")
        print(f"    Stim detectors: {self.num_stim_detectors}")
        print(f"    Graph nodes: {self.graph_mapper.num_nodes}")
        print(f"    Graph feature dim: 6 (1 syndrome + 3 color + 2 coord)")

    def _create_stim_reference(self):
        """Phase 1과 동일한 Stim 회로를 생성합니다."""
        try:
            from generators.color_code import create_color_code_circuit
        except ImportError:
            try:
                from simulation.generators.color_code import create_color_code_circuit
            except ImportError:
                import stim
                return stim.Circuit.generated(
                    "color_code:memory_xyz",
                    distance=self.distance,
                    rounds=self.num_rounds,
                    before_round_data_depolarization=0.001
                )

        return create_color_code_circuit(self.distance, self.num_rounds, 0.001)

    def _initialize_mappers(self):
        """Stim 매퍼를 초기화합니다."""
        from common.mapper_colorcode import ColorCodeGraphMapper, ColorCodeImageMapper

        x_stabs = [[0,1,2,3], [1,2,4,5], [2,3,5,6]]
        z_stabs = [[0,1,2,3], [1,2,4,5], [2,3,5,6]]
        self.graph_mapper = ColorCodeGraphMapper(self.distance, self.num_rounds, x_stabs, z_stabs)
        self.image_mapper = ColorCodeImageMapper(self.distance, self.num_rounds, 6)
        self.edge_index = self.graph_mapper.get_edges()

        # Phase 1 학습에 사용된 엣지 파일이 있으면 우선 사용
        if self.edge_dir is not None:
            edge_path = os.path.join(self.edge_dir, f"edges_d{self.distance}.npy")
        else:
            # fallback: 기존 기본 경로
            edge_path = os.path.join(
                stim_dir, "dataset", "color_code", "graph",
                f"edges_d{self.distance}.npy"
            )

        if os.path.exists(edge_path):
            self.edge_index = np.load(edge_path)
            print(f"    Loaded Phase 1 edges from: {edge_path}")
        else:
            print(f"    [Warning] Edge file not found: {edge_path}")
            print(f"    Using edges from SyndromeGraphMapper instead.")

    def ionq_to_stim_detectors(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        IonQ 원시 앵실러 측정값을 Stim-style 디텍터 배열로 변환합니다.

        Stim 디텍터 = 연속 라운드 간의 XOR (temporal difference)
        - Round 0: detector = measurement (초기 상태가 |0⟩이므로 XOR 0 = 그대로)
        - Round r (r>0): detector = measurement[r] XOR measurement[r-1]

        Args:
            raw_syndromes: (N, num_rounds, num_stabilizers_per_round)

        Returns:
            stim_detectors: (N, num_stim_detectors) Stim 호환 1D 디텍터 배열
        """
        N, num_rounds, num_stab = raw_syndromes.shape

        detectors = np.zeros((N, num_rounds, num_stab), dtype=np.float32)

        # Round 0: 그대로 (초기 |0⟩ 대비)
        detectors[:, 0, :] = raw_syndromes[:, 0, :]

        # Round r>0: 연속 차분 (temporal difference)
        for r in range(1, num_rounds):
            detectors[:, r, :] = (raw_syndromes[:, r, :] != raw_syndromes[:, r - 1, :]).astype(np.float32)

        # Flatten: (N, num_rounds * num_stab)
        flat_detectors = detectors.reshape(N, -1)

        # Stim 디텍터 수와 맞추기
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
        """
        IonQ 신드롬 → Graph 모델 입력 형태로 변환합니다.

        Returns:
            node_features: (N, num_nodes, feature_dim)
            edge_index: (2, num_edges)
        """
        stim_detectors = self.ionq_to_stim_detectors(raw_syndromes)
        node_features = self.graph_mapper.map_to_node_features(stim_detectors)
        return node_features, self.edge_index

    def to_image_format(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        IonQ 신드롬 → Image 모델 입력 형태로 변환합니다.

        Returns:
            images: (N, 1, H, W)
        """
        stim_detectors = self.ionq_to_stim_detectors(raw_syndromes)
        images = self.image_mapper.map_to_images(stim_detectors)
        return images

    def get_model_input_shape(self, model_type: str) -> tuple:
        """모델 타입에 따른 예상 입력 shape을 반환합니다."""
        if model_type == "graph":
            return (self.graph_mapper.num_nodes, 6)
        elif model_type == "image":
            return (1, self.image_mapper.height, self.image_mapper.width)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


# ==============================================================================
# 단독 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    print("=== StimFormatConverter Test ===")

    converter = StimFormatConverter(distance=3, num_rounds=3)

    fake_syndromes = np.random.randint(0, 2, size=(5, 3, 6)).astype(np.float32)

    graph_features, edges = converter.to_graph_format(fake_syndromes)
    print(f"\nGraph format: {graph_features.shape}")
    print(f"Edges: {edges.shape}")

    images = converter.to_image_format(fake_syndromes)
    print(f"Image format: {images.shape}")

    print(f"\nExpected graph input: {converter.get_model_input_shape('graph')}")
    print(f"Expected image input: {converter.get_model_input_shape('image')}")