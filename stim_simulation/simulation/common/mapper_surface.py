"""
Surface Code 전용 그래프 매퍼

Stim의 DEM 기반 매퍼 대신, 하드웨어(IBM)와 동일한 stabilizer 구조를 사용합니다.
각 노드는 (stabilizer, round) 쌍이며, 엣지는 공간적/시간적 인접성으로 정의됩니다.

노드 특성 (6차원):
  [0] syndrome value (0 or 1)
  [1] stabilizer type: X=0, Z=1
  [2] normalized spatial x coordinate
  [3] normalized spatial y coordinate
  [4] normalized temporal coordinate (round / num_rounds)
  [5] boundary flag: bulk=0, boundary=1

엣지:
  - 공간: 같은 라운드 내에서 data qubit을 공유하는 stabilizer 쌍
  - 시간: 같은 stabilizer의 인접 라운드 간 연결
"""

import numpy as np
from typing import List, Dict, Tuple


class SurfaceCodeGraphMapper:
    """
    Surface Code 전용 그래프 매퍼.
    
    하드웨어의 stabilizer 측정 구조와 1:1 대응되는 그래프를 생성합니다.
    distance에 관계없이 동적으로 동작합니다.
    """

    def __init__(self, distance: int, num_rounds: int,
                 x_stabilizers: List[List[int]], z_stabilizers: List[List[int]]):
        """
        Args:
            distance: Code distance
            num_rounds: QEC 라운드 수
            x_stabilizers: X-type stabilizer들의 data qubit 인덱스 리스트
            z_stabilizers: Z-type stabilizer들의 data qubit 인덱스 리스트
        """
        self.distance = distance
        self.num_rounds = num_rounds
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers

        # 전체 stabilizer 리스트: X먼저, Z다음 (Qiskit 회로 순서와 동일)
        self.all_stabilizers = x_stabilizers + z_stabilizers
        self.num_stabilizers = len(self.all_stabilizers)
        self.num_nodes = self.num_stabilizers * num_rounds

        # stabilizer 타입: 0=X, 1=Z
        self.stab_types = [0] * len(x_stabilizers) + [1] * len(z_stabilizers)

        # 공간 좌표 계산 (data qubit 중심)
        self.stab_coords = self._compute_stabilizer_coords()

        # boundary 여부 (weight-2 stabilizer = boundary)
        self.is_boundary = [1 if len(s) == 2 else 0 for s in self.all_stabilizers]

        # 엣지 생성
        self.edge_index = self._build_edges()

        print(f"[SurfaceCodeGraphMapper] d={distance}, rounds={num_rounds}")
        print(f"    Stabilizers: {self.num_stabilizers} ({len(x_stabilizers)} X + {len(z_stabilizers)} Z)")
        print(f"    Nodes: {self.num_nodes}, Edges: {self.edge_index.shape[1]}")

    def _compute_stabilizer_coords(self) -> List[Tuple[float, float]]:
        """
        각 stabilizer의 공간 좌표를 data qubit 위치의 중심으로 계산합니다.
        Data qubit은 distance × distance 격자에 배치됩니다.
        """
        d = self.distance
        coords = []
        for stab in self.all_stabilizers:
            # data qubit index → (row, col) in d×d grid
            rows = [q // d for q in stab]
            cols = [q % d for q in stab]
            center_r = sum(rows) / len(rows)
            center_c = sum(cols) / len(cols)
            coords.append((center_r, center_c))
        return coords

    def _build_edges(self) -> np.ndarray:
        """
        공간적 + 시간적 엣지를 생성합니다.
        
        공간 엣지: 같은 라운드 내에서 data qubit을 공유하는 stabilizer 쌍
        시간 엣지: 같은 stabilizer의 인접 라운드 간 연결
        """
        edges = set()

        # 1. 공간 엣지: data qubit 공유 여부
        spatial_neighbors = []
        for i in range(self.num_stabilizers):
            for j in range(i + 1, self.num_stabilizers):
                shared = set(self.all_stabilizers[i]) & set(self.all_stabilizers[j])
                if shared:
                    spatial_neighbors.append((i, j))

        # 각 라운드에 공간 엣지 적용
        for r in range(self.num_rounds):
            for i, j in spatial_neighbors:
                node_i = r * self.num_stabilizers + i
                node_j = r * self.num_stabilizers + j
                edges.add((node_i, node_j))
                edges.add((node_j, node_i))

        # 2. 시간 엣지: 같은 stabilizer의 인접 라운드
        for r in range(self.num_rounds - 1):
            for s in range(self.num_stabilizers):
                node_curr = r * self.num_stabilizers + s
                node_next = (r + 1) * self.num_stabilizers + s
                edges.add((node_curr, node_next))
                edges.add((node_next, node_curr))

        if not edges:
            return np.zeros((2, 0), dtype=np.int64)

        edge_array = np.array(sorted(edges), dtype=np.int64).T
        return edge_array

    def map_to_node_features(self, flat_detectors: np.ndarray) -> np.ndarray:
        """
        1D detector 배열 → 노드 특성 배열로 변환합니다.

        Args:
            flat_detectors: (N, num_stabilizers * num_rounds)
                            순서: [round0_stab0, round0_stab1, ..., round0_stabK,
                                   round1_stab0, ..., roundR_stabK]

        Returns:
            node_features: (N, num_nodes, 6)
        """
        N = flat_detectors.shape[0]
        features = np.zeros((N, self.num_nodes, 6), dtype=np.float32)

        # 좌표 정규화
        if self.stab_coords:
            max_coord = max(max(c) for c in self.stab_coords) if self.stab_coords else 1.0
            if max_coord == 0:
                max_coord = 1.0
        else:
            max_coord = 1.0

        for r in range(self.num_rounds):
            for s in range(self.num_stabilizers):
                node_idx = r * self.num_stabilizers + s

                # [0] syndrome value
                features[:, node_idx, 0] = flat_detectors[:, r * self.num_stabilizers + s]

                # [1] stabilizer type: X=0, Z=1
                features[:, node_idx, 1] = self.stab_types[s]

                # [2] normalized spatial x
                features[:, node_idx, 2] = self.stab_coords[s][0] / max_coord

                # [3] normalized spatial y
                features[:, node_idx, 3] = self.stab_coords[s][1] / max_coord

                # [4] normalized temporal coordinate
                features[:, node_idx, 4] = r / max(1, self.num_rounds - 1)

                # [5] boundary flag
                features[:, node_idx, 5] = self.is_boundary[s]

        return features

    def get_edges(self) -> np.ndarray:
        """엣지 인덱스를 반환합니다."""
        return self.edge_index


class SurfaceCodeImageMapper:
    """
    Surface Code 전용 이미지 매퍼.
    
    stabilizer 측정값을 2D 이미지로 변환합니다.
    """

    def __init__(self, distance: int, num_rounds: int, num_stabilizers: int):
        self.distance = distance
        self.num_rounds = num_rounds
        self.num_stabilizers = num_stabilizers

        # 이미지 크기: (2d-1) × (2d-1) 격자에 stabilizer를 배치
        self.height = 2 * distance - 1
        self.width = 2 * distance - 1

        print(f"[SurfaceCodeImageMapper] Grid: {self.height}x{self.width}")

    def map_to_images(self, flat_detectors: np.ndarray) -> np.ndarray:
        """
        1D detector → 2D 이미지로 변환합니다.
        
        각 라운드의 stabilizer 값을 (2d-1)×(2d-1) 격자에 배치합니다.
        채널 수 = num_rounds

        Args:
            flat_detectors: (N, num_stabilizers * num_rounds)

        Returns:
            images: (N, num_rounds, height, width)
        """
        N = flat_detectors.shape[0]
        images = np.zeros((N, self.num_rounds, self.height, self.width), dtype=np.float32)

        detectors_3d = flat_detectors.reshape(N, self.num_rounds, self.num_stabilizers)

        # stabilizer를 격자에 배치
        # Surface code (2d-1)×(2d-1) 격자에서:
        #   (even, even) = data qubit
        #   (even, odd) 또는 (odd, even) = ancilla
        stab_idx = 0
        for r_grid in range(self.height):
            for c_grid in range(self.width):
                if (r_grid + c_grid) % 2 == 1:  # ancilla 위치
                    if stab_idx < self.num_stabilizers:
                        for t in range(self.num_rounds):
                            images[:, t, r_grid, c_grid] = detectors_3d[:, t, stab_idx]
                        stab_idx += 1

        return images