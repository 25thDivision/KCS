"""
Color Code 전용 그래프/이미지 매퍼

Qiskit/IonQ의 6-ancilla 구조와 동일한 노드/엣지를 생성합니다.
각 노드는 (stabilizer, round) 쌍입니다.

노드 특성 (6차원):
  [0] syndrome value (0 or 1)
  [1] stabilizer type: X=0, Z=1
  [2] normalized spatial x coordinate
  [3] normalized spatial y coordinate
  [4] normalized temporal coordinate
  [5] color: R=0, G=0.5, B=1.0

엣지:
  - 공간: 같은 라운드 내에서 data qubit을 공유하는 stabilizer 쌍
  - 시간: 같은 stabilizer의 인접 라운드 간 연결
"""

import numpy as np
from typing import List, Tuple


# Color code d=3 data qubit 좌표 (삼각형 격자)
COLORCODE_D3_DATA_COORDS = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (2.0, 0.0),
    3: (0.5, 1.0),
    4: (1.5, 1.0),
    5: (2.5, 1.0),
    6: (1.0, 2.0),
}


class ColorCodeGraphMapper:
    def __init__(self, distance: int, num_rounds: int,
                 x_stabilizers: List[List[int]], z_stabilizers: List[List[int]],
                 colors: List[str] = None):
        """
        Args:
            distance: Code distance
            num_rounds: QEC 라운드 수
            x_stabilizers: X-type stabilizer들의 data qubit 리스트
            z_stabilizers: Z-type stabilizer들의 data qubit 리스트
            colors: 각 stabilizer 색 (X, Z 순서). None이면 자동 할당
        """
        self.distance = distance
        self.num_rounds = num_rounds
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers
        self.all_stabilizers = x_stabilizers + z_stabilizers
        self.num_stabilizers = len(self.all_stabilizers)
        self.num_nodes = self.num_stabilizers * num_rounds

        # stabilizer 타입: 0=X, 1=Z
        self.stab_types = [0] * len(x_stabilizers) + [1] * len(z_stabilizers)

        # 색상: R=0, G=0.5, B=1.0 (X 3개 + Z 3개 = R,G,B,R,G,B)
        if colors:
            color_map = {"R": 0.0, "G": 0.5, "B": 1.0}
            self.stab_colors = [color_map.get(c, 0.0) for c in colors]
        else:
            self.stab_colors = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]  # R,G,B,R,G,B

        # 공간 좌표 계산
        self.stab_coords = self._compute_stabilizer_coords()

        # 엣지 생성
        self.edge_index = self._build_edges()

        print(f"[ColorCodeGraphMapper] d={distance}, rounds={num_rounds}")
        print(f"    Stabilizers: {self.num_stabilizers} ({len(x_stabilizers)} X + {len(z_stabilizers)} Z)")
        print(f"    Nodes: {self.num_nodes}, Edges: {self.edge_index.shape[1]}")

    def _compute_stabilizer_coords(self) -> List[Tuple[float, float]]:
        """각 stabilizer의 공간 좌표를 data qubit 중심으로 계산합니다."""
        data_coords = COLORCODE_D3_DATA_COORDS
        coords = []
        for stab in self.all_stabilizers:
            xs = [data_coords[q][0] for q in stab]
            ys = [data_coords[q][1] for q in stab]
            coords.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        return coords

    def _build_edges(self) -> np.ndarray:
        """공간적 + 시간적 엣지를 생성합니다."""
        edges = set()

        # 공간 엣지: data qubit 공유
        spatial_neighbors = []
        for i in range(self.num_stabilizers):
            for j in range(i + 1, self.num_stabilizers):
                shared = set(self.all_stabilizers[i]) & set(self.all_stabilizers[j])
                if shared:
                    spatial_neighbors.append((i, j))

        for r in range(self.num_rounds):
            for i, j in spatial_neighbors:
                node_i = r * self.num_stabilizers + i
                node_j = r * self.num_stabilizers + j
                edges.add((node_i, node_j))
                edges.add((node_j, node_i))

        # 시간 엣지
        for r in range(self.num_rounds - 1):
            for s in range(self.num_stabilizers):
                node_curr = r * self.num_stabilizers + s
                node_next = (r + 1) * self.num_stabilizers + s
                edges.add((node_curr, node_next))
                edges.add((node_next, node_curr))

        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(sorted(edges), dtype=np.int64).T

    def map_to_node_features(self, flat_detectors: np.ndarray) -> np.ndarray:
        """
        1D detector 배열 → 노드 특성 배열.

        Args:
            flat_detectors: (N, num_stabilizers * num_rounds)

        Returns:
            node_features: (N, num_nodes, 6)
        """
        N = flat_detectors.shape[0]
        features = np.zeros((N, self.num_nodes, 6), dtype=np.float32)

        max_coord = max(max(abs(c[0]), abs(c[1])) for c in self.stab_coords)
        if max_coord == 0:
            max_coord = 1.0

        for r in range(self.num_rounds):
            for s in range(self.num_stabilizers):
                node_idx = r * self.num_stabilizers + s
                features[:, node_idx, 0] = flat_detectors[:, r * self.num_stabilizers + s]
                features[:, node_idx, 1] = self.stab_types[s]
                features[:, node_idx, 2] = self.stab_coords[s][0] / max_coord
                features[:, node_idx, 3] = self.stab_coords[s][1] / max_coord
                features[:, node_idx, 4] = r / max(1, self.num_rounds - 1)
                features[:, node_idx, 5] = self.stab_colors[s]
        return features

    def get_edges(self) -> np.ndarray:
        return self.edge_index


class ColorCodeImageMapper:
    def __init__(self, distance: int, num_rounds: int, num_stabilizers: int):
        self.distance = distance
        self.num_rounds = num_rounds
        self.num_stabilizers = num_stabilizers
        # Color code d=3: 3x2 grid (넉넉하게)
        self.height = 3
        self.width = 2
        print(f"[ColorCodeImageMapper] Grid: {self.height}x{self.width}")

    def map_to_images(self, flat_detectors: np.ndarray) -> np.ndarray:
        """
        1D detector → 2D 이미지.
        6 stabilizers를 3×2 격자에 배치 (X: 왼쪽 열, Z: 오른쪽 열).
        채널 수 = num_rounds.
        """
        N = flat_detectors.shape[0]
        images = np.zeros((N, self.num_rounds, self.height, self.width), dtype=np.float32)
        detectors_3d = flat_detectors.reshape(N, self.num_rounds, self.num_stabilizers)

        # 배치: stabilizer 0-2 (X R,G,B) → col 0, stabilizer 3-5 (Z R,G,B) → col 1
        for t in range(self.num_rounds):
            for s in range(min(3, self.num_stabilizers)):
                images[:, t, s, 0] = detectors_3d[:, t, s]      # X-type
            for s in range(3, min(6, self.num_stabilizers)):
                images[:, t, s - 3, 1] = detectors_3d[:, t, s]  # Z-type

        return images