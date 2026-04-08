"""
Color Code 전용 그래프/이미지 매퍼 (d=3, 5, 7 지원)

Qiskit/IonQ의 개별 stabilizer 구조와 동일한 노드/엣지를 생성합니다.

노드 특성 (6차원):
  [0] syndrome value (0 or 1)
  [1] stabilizer type: X=0, Z=1
  [2] normalized spatial x coordinate
  [3] normalized spatial y coordinate
  [4] normalized temporal coordinate
  [5] color: face index normalized (0 ~ 1)
"""

import numpy as np
from typing import List, Tuple


class ColorCodeGraphMapper:
    def __init__(self, distance: int, num_rounds: int,
                 x_stabilizers: List[List[int]], z_stabilizers: List[List[int]],
                 data_coords: dict = None):
        """
        Args:
            distance: Code distance
            num_rounds: QEC 라운드 수
            x_stabilizers: X-type stabilizer들의 data qubit 리스트
            z_stabilizers: Z-type stabilizer들의 data qubit 리스트
            data_coords: {local_idx: (x, y)} 좌표 딕셔너리. None이면 자동 생성.
        """
        self.distance = distance
        self.num_rounds = num_rounds
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers
        self.all_stabilizers = x_stabilizers + z_stabilizers
        self.num_stabilizers = len(self.all_stabilizers)
        self.num_nodes = self.num_stabilizers * num_rounds

        self.stab_types = [0] * len(x_stabilizers) + [1] * len(z_stabilizers)

        # 색상: face index 기반 정규화
        num_faces = len(x_stabilizers)
        face_colors = [i / max(1, num_faces - 1) for i in range(num_faces)]
        self.stab_colors = face_colors + face_colors  # X faces + Z faces

        # 좌표
        if data_coords:
            self.data_coords = data_coords
        else:
            from generators.color_code import COLORCODE_DATA_COORDS
            self.data_coords = COLORCODE_DATA_COORDS.get(distance, {})

        self.stab_coords = self._compute_stabilizer_coords()
        self.edge_index = self._build_edges()

        print(f"[ColorCodeGraphMapper] d={distance}, rounds={num_rounds}")
        print(f"    Stabilizers: {self.num_stabilizers} ({len(x_stabilizers)} X + {len(z_stabilizers)} Z)")
        print(f"    Nodes: {self.num_nodes}, Edges: {self.edge_index.shape[1]}")

    def _compute_stabilizer_coords(self) -> List[Tuple[float, float]]:
        coords = []
        for stab in self.all_stabilizers:
            if self.data_coords:
                xs = [self.data_coords[q][0] for q in stab if q in self.data_coords]
                ys = [self.data_coords[q][1] for q in stab if q in self.data_coords]
                if xs and ys:
                    coords.append((sum(xs) / len(xs), sum(ys) / len(ys)))
                else:
                    coords.append((0.0, 0.0))
            else:
                coords.append((0.0, 0.0))
        return coords

    def _build_edges(self) -> np.ndarray:
        edges = set()

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
        N = flat_detectors.shape[0]
        features = np.zeros((N, self.num_nodes, 6), dtype=np.float32)

        all_vals = [abs(c[0]) for c in self.stab_coords] + [abs(c[1]) for c in self.stab_coords]
        max_coord = max(all_vals) if all_vals else 1.0
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
        self.num_faces = num_stabilizers // 2

        # 이미지 크기: faces_per_type 행 × 2 열 (X | Z)
        self.height = self.num_faces
        self.width = 2
        print(f"[ColorCodeImageMapper] Grid: {self.height}x{self.width}, faces={self.num_faces}")

    def map_to_images(self, flat_detectors: np.ndarray) -> np.ndarray:
        N = flat_detectors.shape[0]
        images = np.zeros((N, self.num_rounds, self.height, self.width), dtype=np.float32)
        detectors_3d = flat_detectors.reshape(N, self.num_rounds, self.num_stabilizers)

        for t in range(self.num_rounds):
            # X-type (앞쪽 num_faces개) → col 0
            for s in range(self.num_faces):
                images[:, t, s, 0] = detectors_3d[:, t, s]
            # Z-type (뒤쪽 num_faces개) → col 1
            for s in range(self.num_faces):
                images[:, t, s, 1] = detectors_3d[:, t, self.num_faces + s]

        return images