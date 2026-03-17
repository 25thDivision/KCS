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

# Stim 시뮬레이션 경로 추가
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
    
    Stim의 SyndromeGraphMapper / SyndromeImageMapper를 내부적으로 사용하며,
    IonQ 원시 측정값을 Stim 디텍터 형식으로 변환한 후 매퍼에 전달합니다.
    
    Parameters:
        distance (int): Code distance
        num_rounds (int): QEC 라운드 수 (Phase 1 학습 시와 동일해야 함)
    """
    
    def __init__(self, distance: int, num_rounds: int):
        self.distance = distance
        self.num_rounds = num_rounds
        
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
        from common.mapper_graph import SyndromeGraphMapper
        from common.mapper_image import SyndromeImageMapper
        
        self.graph_mapper = SyndromeGraphMapper(self.stim_circuit)
        self.image_mapper = SyndromeImageMapper(self.stim_circuit)
        self.edge_index = self.graph_mapper.get_edges()
        
        # Phase 1 학습에 사용된 엣지 파일이 있으면 그걸 우선 사용
        edge_path = os.path.join(
            stim_dir, "dataset", "color_code", "graph", 
            f"edges_d{self.distance}.npy"
        )
        if os.path.exists(edge_path):
            self.edge_index = np.load(edge_path)
            print(f"    Loaded Phase 1 edges from: {edge_path}")
    
    def ionq_to_stim_detectors(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        IonQ 원시 앵실러 측정값을 Stim-style 디텍터 배열로 변환합니다.
        
        Stim 디텍터 = 연속 라운드 간의 XOR (temporal difference)
        - Round 0: detector = measurement (초기 상태가 |0⟩이므로 XOR 0 = 그대로)
        - Round r (r>0): detector = measurement[r] XOR measurement[r-1]
        
        Args:
            raw_syndromes: (N, num_rounds, num_stabilizers_per_round)
                           IonQ에서 추출한 라운드별 원시 앵실러 측정값
        
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
            # 부족하면 0으로 패딩
            padded = np.zeros((N, self.num_stim_detectors), dtype=np.float32)
            padded[:, :flat_detectors.shape[1]] = flat_detectors
            flat_detectors = padded
            print(f"    [Warning] Padded {self.num_stim_detectors - N} extra detector positions with 0")
        elif flat_detectors.shape[1] > self.num_stim_detectors:
            # 초과하면 자르기
            flat_detectors = flat_detectors[:, :self.num_stim_detectors]
            print(f"    [Warning] Truncated to {self.num_stim_detectors} detectors")
        
        return flat_detectors
    
    def to_graph_format(self, raw_syndromes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        IonQ 신드롬 → Graph 모델 입력 형태로 변환합니다.
        
        Args:
            raw_syndromes: (N, num_rounds, num_stabilizers_per_round)
        
        Returns:
            node_features: (N, num_nodes, feature_dim) — Phase 1 모델 입력 호환
            edge_index: (2, num_edges)
        """
        stim_detectors = self.ionq_to_stim_detectors(raw_syndromes)
        node_features = self.graph_mapper.map_to_node_features(stim_detectors)
        return node_features, self.edge_index
    
    def to_image_format(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        IonQ 신드롬 → Image 모델 입력 형태로 변환합니다.
        
        Args:
            raw_syndromes: (N, num_rounds, num_stabilizers_per_round)
        
        Returns:
            images: (N, 1, H, W) — Phase 1 CNN/UNet 모델 입력 호환
        """
        stim_detectors = self.ionq_to_stim_detectors(raw_syndromes)
        images = self.image_mapper.map_to_images(stim_detectors)
        return images
    
    def get_model_input_shape(self, model_type: str) -> tuple:
        """모델 타입에 따른 예상 입력 shape을 반환합니다."""
        if model_type == "graph":
            return (self.graph_mapper.num_nodes, 6)  # (num_nodes, feature_dim)
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
    
    # 가짜 IonQ 신드롬 (3 rounds, 6 stabilizers)
    fake_syndromes = np.random.randint(0, 2, size=(5, 3, 6)).astype(np.float32)
    
    # Graph 형태 변환 테스트
    graph_features, edges = converter.to_graph_format(fake_syndromes)
    print(f"\nGraph format: {graph_features.shape}")
    print(f"Edges: {edges.shape}")
    
    # Image 형태 변환 테스트
    images = converter.to_image_format(fake_syndromes)
    print(f"Image format: {images.shape}")
    
    print(f"\nExpected graph input: {converter.get_model_input_shape('graph')}")
    print(f"Expected image input: {converter.get_model_input_shape('image')}")
