import stim
import numpy as np
import networkx as nx  # 그래프 컬러링을 위해 networkx 라이브러리가 필요합니다
from typing import Tuple, Set

class SyndromeGraphMapper:
    """
    Stim 회로 데이터를 GNN 학습용 그래프 구조로 변환하는 클래스입니다
    탐지기(Detector)를 노드로, 에러 상관관계를 엣지로 정의합니다
    """

    def __init__(self, circuit: stim.Circuit):
        """
        매퍼를 초기화하고 정적 그래프 구조를 생성합니다

        Args:
            circuit (stim.Circuit): 분석할 Stim 회로 객체입니다
        """
        # 1. 탐지기 좌표 및 인덱스를 추출합니다
        self.coords_dict = circuit.get_detector_coordinates()
        self.node_indices = np.array(sorted(list(self.coords_dict.keys())))
        self.num_nodes = len(self.node_indices)
        
        # 유효한 인덱스인지 빠르게 확인하기 위해 집합(Set)으로 저장합니다
        self.valid_indices_set = set(self.node_indices)

        # 2. DEM(Detector Error Model)을 분석하여 그래프 엣지를 생성합니다
        print("[SyndromeGraphMapper] Building graph edges from DEM... (this may take a moment)")
        self.edge_index = self._build_edges_from_dem(circuit)
        
        # 3. [NEW] 그래프 컬러링을 통해 Stabilizer Type(색깔)을 할당합니다
        print("[SyndromeGraphMapper] Assigning stabilizer types (colors)...")
        self.node_types = self._assign_stabilizer_types()

        print(f"[SyndromeGraphMapper] Initialized. Nodes: {self.num_nodes}, Types: {len(np.unique(self.node_types))}")

    def _build_edges_from_dem(self, circuit: stim.Circuit) -> np.ndarray:
        """
        DEM을 분석하여 상관관계가 있는 탐지기 사이에 엣지를 연결합니다
        
        Returns:
            np.ndarray: [2, num_edges] 크기의 엣지 리스트입니다 (Source, Target)
        """
        dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        edges = set()

        for instruction in dem.flattened():
            # 에러 명령(instruction)만 분석합니다
            if instruction.type == "error":
                # 해당 에러가 트리거하는 탐지기 목록을 가져옵니다
                dets = [t.val for t in instruction.targets_copy() if t.is_relative_detector_id()]
                
                # 하나의 에러가 2개 이상의 탐지기를 건드리면, 그 탐지기들은 서로 연결된 것입니다
                for i in range(len(dets)):
                    for j in range(i + 1, len(dets)):
                        u, v = dets[i], dets[j]
                        if u in self.valid_indices_set and v in self.valid_indices_set:
                            # 무방향 그래프이므로 양방향 모두 추가합니다
                            edges.add((u, v))
                            edges.add((v, u))

        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
            
        return np.array(list(edges)).T.astype(np.int64)

    def _assign_stabilizer_types(self) -> np.ndarray:
        """
        [NEW] 그래프 컬러링 알고리즘을 사용하여 각 노드(탐지기)의 타입(0, 1, 2)을 자동으로 할당합니다
        Color Code의 신드롬 그래프는 3-colorable 특성을 가지기 때문입니다
        """
        # 1. NetworkX 그래프 객체를 생성합니다
        G = nx.Graph()
        G.add_nodes_from(self.node_indices)
        if self.edge_index.shape[1] > 0:
            edges = self.edge_index.T.tolist()
            G.add_edges_from(edges)
        
        # 2. Greedy Coloring을 수행합니다
        # 'largest_first' 전략은 차수가 높은 노드부터 색칠하므로 안정적인 결과를 제공합니다
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        
        # 3. 결과를 배열로 변환합니다 (노드 인덱스 순서를 맞춥니다)
        types = np.zeros(self.num_nodes, dtype=np.int64)
        for i, node_idx in enumerate(self.node_indices):
            # 고립된 노드의 경우 기본값 0을 부여합니다
            color = coloring.get(node_idx, 0)
            types[i] = color % 3 # 타입을 3가지(0, 1, 2)로 제한합니다
            
        return types

    def map_to_node_features(self, detector_data: np.ndarray) -> np.ndarray:
        """
        [UPDATED] 신드롬 + Stabilizer Type + 좌표(Coordinate) 정보를 모두 포함합니다.
        Input Feature Dim: 1 (Syndrome) + 3 (Color) + 2 (Coord) = 6
        """
        shots = detector_data.shape[0]
        
        # 1. Syndrome Feature (Shape: [Batch, N, 1])
        syndrome_feat = np.zeros((shots, self.num_nodes, 1), dtype=np.float32)
        if detector_data.shape[1] >= self.num_nodes:
            syndrome_feat[:, :, 0] = detector_data[:, self.node_indices]
            
        # 2. Stabilizer Type Feature (Shape: [Batch, N, 3])
        type_feat = np.zeros((shots, self.num_nodes, 3), dtype=np.float32)
        type_one_hot = np.eye(3)[self.node_types]
        type_feat[:] = type_one_hot
        
        # 3. [NEW] Coordinate Feature (Shape: [Batch, N, 2])
        # 좌표 정보를 추가하여 GNN이 노드 간의 상대적 위치를 학습하게 돕습니다.
        coord_feat = np.zeros((shots, self.num_nodes, 2), dtype=np.float32)
        
        # 저장해둔 좌표 딕셔너리에서 (x, y)를 꺼내 정규화(Normalize)하여 넣습니다.
        # 정규화 안 하면 좌표값이 너무 커서 학습이 불안정해질 수 있습니다.
        coords_list = []
        for node_idx in self.node_indices:
            # stim 좌표는 리스트 형태 [x, y, z...]일 수 있음. 앞 2개만 사용
            c = self.coords_dict.get(node_idx, [0, 0])[:2]
            coords_list.append(c)
        
        coords_array = np.array(coords_list, dtype=np.float32)
        
        # 최대 좌표값으로 나누어 0~1 사이로 정규화 (선택 사항이지만 권장)
        max_val = np.max(np.abs(coords_array)) if len(coords_array) > 0 else 1.0
        if max_val > 0:
            coords_array /= max_val
            
        coord_feat[:] = coords_array
        
        # 4. 모든 피처 연결 (Concatenate)
        # 최종 Shape: [Batch, Num_Nodes, 6]
        return np.concatenate([syndrome_feat, type_feat, coord_feat], axis=2)

    def get_edges(self) -> np.ndarray:
        """
        생성된 엣지 리스트를 반환합니다
        """
        return self.edge_index