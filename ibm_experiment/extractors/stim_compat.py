"""
IBM 신드롬 → Stim 호환 형태 변환기

코드 타입별 변환 경로:
  - heavyhex_surface_code (legacy): depth7 heavy-hex syndrome은 HW가 측정하는
    연산자가 Stim의 TRUE stabilizer와 일치하지 않아 `reorder_hw_to_stim()`으로
    Z-stab만 매핑하고 X-stab은 0으로 마스킹한다.
  - surface_code: rotated surface code의 Qiskit 회로는 Stim의 기본
    `rotated_memory_z`와 stabilizer 지지대(support)가 맞지 않는다 (X↔Z label
    swap, logical observable 방향이 다름). 따라서 우리 Qiskit 회로와 정확히
    동일한 measurement schedule을 가진 Stim 참조 회로를 직접 구성하고,
    `compile_m2d_converter()`로 HW measurement → detector 변환을 수행한다.
    (Round 0에서는 Z-stab detector만 존재하고, round r≥1은 이전 round와의
    XOR, 최종은 data 측정에서 재구성한 Z-stab과 마지막 syndrome의 XOR.)

Ancilla reuse (no reset) 대응:
  Nighthawk/ibm_miami에는 reset 명령이 ISA에 없으므로 Qiskit HW 회로는
  round 간 ancilla를 reset하지 않고 재사용한다. 후속 round의 CX 체인이
  측정 후 classical collapse된 ancilla 상태 위로 새 stabilizer를 XOR
  시키므로 raw 측정값은 stabilizer의 *누적 XOR* (raw_r = s_1⊕…⊕s_r) 이
  된다. Stim 참조 회로는 그대로 `MR` (measure-reset) 을 유지하고,
  HW 측에서 한 단계 single-differencing (raw_r ⊕ raw_{r-1} = s_r) 을 적용해
  per-round syndrome으로 환산한 뒤 downstream 로직에 넘긴다. 이렇게 하면
  DEM, m2d converter, ML 모델 모두 기존 MR 가정의 입력을 그대로 받는다.

공통 처리:
  1. ML 경로용: cumulative→per-round 변환 후 temporal differencing된 detector
  2. MWPM 경로용: cumulative→per-round 변환 후 Stim compile_m2d_converter 결과
     (surface_code only)
  3. SurfaceCodeGraphMapper / SurfaceCodeImageMapper로 ML 입력 생성
  4. Phase 1에서 저장한 edges_dK.npy가 있으면 로드
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Optional, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
ibm_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(ibm_dir)
stim_dir = os.path.join(root_dir, "stim_simulation")
stim_sim_dir = os.path.join(stim_dir, "simulation")

sys.path.append(stim_dir)
sys.path.append(stim_sim_dir)
sys.path.append(ibm_dir)


# ---------------------------------------------------------------------------
# Stim reference circuit matching the Qiskit SurfaceCodeCircuit schedule.
# ---------------------------------------------------------------------------
def build_qiskit_style_stim_circuit(
    distance: int,
    num_rounds: int,
    x_stabilizers: List[List[int]],
    z_stabilizers: List[List[int]],
    logical_z_qubits: List[int],
    noise: Optional[Dict[str, float]] = None,
):
    """
    Build a stim.Circuit that mirrors the Qiskit SurfaceCodeCircuit memory-Z
    schedule so that detectors / observable match our Qiskit stabilizer set.

    Measurement order (per round): X-stabs then Z-stabs; final data M at end.
    Initial state: |0⟩_L (data and ancilla reset in Z basis).

    Qubit index layout:
        data qubits: 0 .. d²-1
        X ancillas:  d² .. d² + n_x - 1
        Z ancillas:  d² + n_x .. d² + n_x + n_z - 1
    """
    import stim

    d = distance
    num_data = d * d
    n_x = len(x_stabilizers)
    n_z = len(z_stabilizers)
    n_stab = n_x + n_z

    x_anc = list(range(num_data, num_data + n_x))
    z_anc = list(range(num_data + n_x, num_data + n_stab))
    all_anc = x_anc + z_anc  # MR order

    if noise is None:
        noise = {}
    dp = float(noise.get("dp", 0.0))
    mf = float(noise.get("mf", 0.0))
    rf = float(noise.get("rf", 0.0))
    gd = float(noise.get("gd", 0.0))

    c = stim.Circuit()

    # Coordinates for readability only
    for i in range(num_data):
        r, col = divmod(i, d)
        c.append("QUBIT_COORDS", [i], [2 * col + 1, 2 * r + 1])
    for k, stab in enumerate(x_stabilizers):
        xs = [2 * (q % d) + 1 for q in stab]
        ys = [2 * (q // d) + 1 for q in stab]
        c.append("QUBIT_COORDS", [x_anc[k]],
                 [sum(xs) / len(xs), sum(ys) / len(ys)])
    for k, stab in enumerate(z_stabilizers):
        xs = [2 * (q % d) + 1 for q in stab]
        ys = [2 * (q // d) + 1 for q in stab]
        c.append("QUBIT_COORDS", [z_anc[k]],
                 [sum(xs) / len(xs), sum(ys) / len(ys)])

    # Initialize all qubits in |0>
    c.append("R", list(range(num_data)) + all_anc)
    if rf > 0:
        c.append("X_ERROR", list(range(num_data)) + all_anc, rf)
    c.append("TICK")

    for r in range(num_rounds):
        if dp > 0:
            c.append("DEPOLARIZE1", list(range(num_data)), dp)

        # X-stabs: H a; CX a d; H a
        for k, stab in enumerate(x_stabilizers):
            a = x_anc[k]
            c.append("H", [a])
            if gd > 0:
                c.append("DEPOLARIZE1", [a], gd)
            for dq in stab:
                c.append("CX", [a, dq])
                if gd > 0:
                    c.append("DEPOLARIZE2", [a, dq], gd)
            c.append("H", [a])
            if gd > 0:
                c.append("DEPOLARIZE1", [a], gd)

        # Z-stabs: CX d a
        for k, stab in enumerate(z_stabilizers):
            a = z_anc[k]
            for dq in stab:
                c.append("CX", [dq, a])
                if gd > 0:
                    c.append("DEPOLARIZE2", [dq, a], gd)

        if mf > 0:
            c.append("X_ERROR", all_anc, mf)
        c.append("MR", all_anc)
        if rf > 0:
            c.append("X_ERROR", all_anc, rf)

        # DETECTOR lines. rec convention used below (see module docstring):
        #   within a just-completed MR of n_stab measurements in order
        #   [X_0, ..., X_{n_x-1}, Z_0, ..., Z_{n_z-1}], the k-th target is at
        #   rec[-(n_stab - k)]; the same k in the previous round is at
        #   rec[-(2*n_stab - k)].
        if r == 0:
            # |0>_L is not an eigenstate of X-stabs, so only Z-stabs are
            # deterministic in round 0.
            for zk in range(n_z):
                sk = n_x + zk
                c.append("DETECTOR",
                         [stim.target_rec(-(n_stab - sk))],
                         [sk, 0, 0])
        else:
            c.append("SHIFT_COORDS", [], [0, 0, 1])
            for sk in range(n_stab):
                cur = -(n_stab - sk)
                prev = -(2 * n_stab - sk)
                c.append("DETECTOR",
                         [stim.target_rec(cur), stim.target_rec(prev)],
                         [sk, 0, 0])
        c.append("TICK")

    # Final data measurement
    if mf > 0:
        c.append("X_ERROR", list(range(num_data)), mf)
    c.append("M", list(range(num_data)))

    # Reconstruct Z-stabs from data measurements and XOR with last-round Z.
    # After final M (num_data meas), data[i] -> rec[-(num_data - i)].
    # Last round's Z[zk] measurement -> rec[-(num_data + n_z - zk)].
    for zk, stab in enumerate(z_stabilizers):
        targets = []
        for dq in stab:
            targets.append(stim.target_rec(-(num_data - dq)))
        targets.append(stim.target_rec(-(num_data + n_z - zk)))
        c.append("DETECTOR", targets, [n_x + zk, 0, 1])

    # Logical Z observable = XOR of data qubits along the logical-Z support.
    obs = [stim.target_rec(-(num_data - q)) for q in logical_z_qubits]
    c.append("OBSERVABLE_INCLUDE", obs, 0)

    return c


class StimFormatConverter:
    """
    하드웨어 신드롬을 Phase 1 ML 모델 입력 (graph / image) 으로 변환합니다.

    Parameters:
        distance: Code distance
        num_rounds: QEC 라운드 수
        edge_dir: Phase 1 edge 파일 디렉토리 (stim_data_dir)
        code_type: "heavyhex_surface_code" or "surface_code"
    """

    def __init__(self, distance: int, num_rounds: int,
                 edge_dir: str = None, code_type: str = "surface_code"):
        self.distance = distance
        self.num_rounds = num_rounds
        self.edge_dir = edge_dir
        self.code_type = code_type

        if code_type == "heavyhex_surface_code":
            self._init_heavyhex()
        elif code_type == "surface_code":
            self._init_surface_code()
        else:
            raise ValueError(f"Unknown code_type: {code_type}")

        self._initialize_mappers()

        print(f"[StimFormatConverter] Initialized for {code_type} d={distance}, "
              f"rounds={num_rounds}")
        print(f"    Stim detectors: {self.num_stim_detectors}")
        print(f"    Data qubits: {len(self.stim_data_qubit_indices)}")
        print(f"    Graph nodes: {self.graph_mapper.num_nodes}")

    # ------------------------------------------------------------------ #
    # Code-type-specific setup
    # ------------------------------------------------------------------ #
    def _init_heavyhex(self):
        from generators.heavyhex_surface_code import (
            HEAVYHEX_D3, create_heavyhex_surface_code_circuit
        )
        self.cfg = HEAVYHEX_D3
        self.num_stabilizers = self.cfg["num_ancilla"]
        self.num_data_qubits = self.cfg["num_data"]
        self.stim_circuit = create_heavyhex_surface_code_circuit(
            distance=self.distance, rounds=self.num_rounds, noise=0.001
        )
        self.num_stim_detectors = self.num_rounds * self.num_stabilizers
        self.stim_data_qubit_indices = list(range(self.cfg["num_data"]))

        # Mapper inputs: heavyhex has its own ancilla→stabilizer dicts
        z_stabs = [self.cfg["z_stabilizers"][a] for a in self.cfg["z_ancilla"]]
        x_stabs = [self.cfg["x_stabilizers"][a] for a in self.cfg["x_ancilla"]]
        self._mapper_x = z_stabs    # preserving legacy (swapped) ordering
        self._mapper_z = x_stabs

        # Heavyhex has no m2d converter (kept None → MWPM path uses legacy
        # lookup decoder, not Stim DEM).
        self.stim_reference_circuit = None
        self.m2d_converter = None

    def _init_surface_code(self):
        from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
        sc = SurfaceCodeCircuit(distance=self.distance, num_rounds=self.num_rounds)
        self.num_stabilizers = sc.num_stabilizers
        self.num_data_qubits = sc.num_data
        self.stim_data_qubit_indices = list(range(sc.num_data))
        self._mapper_x = sc.x_stabilizers
        self._mapper_z = sc.z_stabilizers
        self.cfg = None
        self.stim_circuit = None

        # Build a Stim reference circuit that mirrors the Qiskit schedule so
        # that compile_m2d_converter produces detectors compatible with our
        # hardware measurement ordering.
        self._x_stabs = sc.x_stabilizers
        self._z_stabs = sc.z_stabilizers
        self._logical_z = sc.logical_z
        self.stim_reference_circuit = build_qiskit_style_stim_circuit(
            distance=self.distance,
            num_rounds=self.num_rounds,
            x_stabilizers=sc.x_stabilizers,
            z_stabilizers=sc.z_stabilizers,
            logical_z_qubits=sc.logical_z,
            noise=None,
        )
        self.m2d_converter = self.stim_reference_circuit.compile_m2d_converter()
        self.num_stim_detectors = self.stim_reference_circuit.num_detectors

    def _initialize_mappers(self):
        from common.mapper_surface import (
            SurfaceCodeGraphMapper, SurfaceCodeImageMapper,
        )
        self.graph_mapper = SurfaceCodeGraphMapper(
            self.distance, self.num_rounds, self._mapper_x, self._mapper_z
        )
        self.image_mapper = SurfaceCodeImageMapper(
            self.distance, self.num_rounds, self.num_stabilizers
        )
        self.edge_index = self.graph_mapper.get_edges()

        # Prefer cached edges from Phase 1 training pipeline
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_data_qubit_indices(self) -> List[int]:
        return self.stim_data_qubit_indices

    def _temporal_differencing_only(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        Legacy ML-format detector conversion: per-round temporal differencing
        with round 0 = raw syndrome. Kept for the ML decoders whose training
        pipeline uses this same representation (stim generate_dataset).

        NOTE: This is NOT valid as Stim DEM detector input (round-0 X-stab
        is not deterministic in |0>_L). Do not feed this to pymatching.
        """
        N, nr, _ = raw_syndromes.shape
        detectors = np.zeros_like(raw_syndromes)
        detectors[:, 0, :] = raw_syndromes[:, 0, :]
        for r in range(1, nr):
            detectors[:, r, :] = (
                raw_syndromes[:, r, :] != raw_syndromes[:, r - 1, :]
            ).astype(np.float32)
        return detectors.reshape(N, -1)

    def _cumulative_to_per_round(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        Convert cumulative-XOR raws from the no-reset Qiskit circuit into
        per-round syndrome values (s_r = raw_r XOR raw_{r-1}, raw_{-1} = 0).

        Applied only for surface_code (heavyhex path keeps its legacy
        handling). After this, downstream treatment matches the MR-based
        Stim reference (raw_r = s_r), so ML temporal-differencing and the
        m2d converter both work unchanged.
        """
        N, nr, ns = raw_syndromes.shape
        per_round = np.zeros_like(raw_syndromes)
        per_round[:, 0, :] = raw_syndromes[:, 0, :]
        for r in range(1, nr):
            per_round[:, r, :] = (
                raw_syndromes[:, r, :] != raw_syndromes[:, r - 1, :]
            ).astype(raw_syndromes.dtype)
        return per_round

    def hw_to_stim_detectors(self, raw_syndromes: np.ndarray) -> np.ndarray:
        """
        HW raw syndrome → ML-format flat detector array.

        Args:
            raw_syndromes: (N, num_rounds, num_stabilizers)

        Returns:
            (N, num_rounds * num_stabilizers)

        Used by the graph/image mappers for ML decoders. MWPM should instead
        go through `hw_to_mwpm_detectors()` which uses the Stim m2d converter.
        """
        if self.code_type == "surface_code":
            raw_syndromes = self._cumulative_to_per_round(raw_syndromes)

        nr = raw_syndromes.shape[1]
        flat = self._temporal_differencing_only(raw_syndromes)

        if self.code_type == "heavyhex_surface_code":
            from generators.heavyhex_surface_code import reorder_hw_to_stim
            flat = reorder_hw_to_stim(flat, nr)

        return flat

    # Alias to keep older call sites working.
    hw_to_stim_syndromes = hw_to_stim_detectors

    def hw_to_mwpm_detectors(self, raw_syndromes: np.ndarray,
                              data_states: np.ndarray) -> np.ndarray:
        """
        HW measurements → Stim DEM-compatible detector array for MWPM.

        Uses the m2d converter built from a Stim reference circuit that
        exactly mirrors our Qiskit measurement schedule:
          chronological measurement order =
            round 0 [X_0..X_{nx-1}, Z_0..Z_{nz-1}],
            round 1 [...], ..., round R-1 [...],
            final M [data_0..data_{nd-1}].

        Args:
            raw_syndromes: (N, num_rounds, num_stabilizers) — same layout as
                           produced by SyndromeExtractor.
            data_states:   (N, num_data) — final data qubit measurements.

        Returns:
            (N, num_stim_detectors) uint8 detector array matching the Stim
            reference circuit's detector order.
        """
        if self.m2d_converter is None:
            raise RuntimeError(
                f"hw_to_mwpm_detectors is only available for surface_code "
                f"(code_type={self.code_type!r})."
            )
        N, nr, ns = raw_syndromes.shape
        if nr != self.num_rounds or ns != self.num_stabilizers:
            raise ValueError(
                f"raw_syndromes shape {(nr, ns)} does not match expected "
                f"({self.num_rounds}, {self.num_stabilizers})."
            )
        if data_states.shape[0] != N or data_states.shape[1] != self.num_data_qubits:
            raise ValueError(
                f"data_states shape {data_states.shape} inconsistent with "
                f"syndromes (N={N}, num_data={self.num_data_qubits})."
            )

        # Qiskit HW circuit has no between-round reset; raws are cumulative.
        # Convert to per-round syndromes so that the MR-based Stim reference
        # sees matching measurement semantics.
        per_round = self._cumulative_to_per_round(raw_syndromes)

        syn_flat = per_round.reshape(N, nr * ns).astype(np.bool_)
        data_flat = data_states.astype(np.bool_)
        measurements = np.concatenate([syn_flat, data_flat], axis=1)

        detectors, _ = self.m2d_converter.convert(
            measurements=measurements, separate_observables=True,
            append_observables=False,
        )
        return detectors.astype(np.uint8)

    def to_graph_format(self, raw_syndromes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        stim_detectors = self.hw_to_stim_detectors(raw_syndromes)
        node_features = self.graph_mapper.map_to_node_features(stim_detectors)
        return node_features, self.edge_index

    def to_image_format(self, raw_syndromes: np.ndarray) -> np.ndarray:
        stim_detectors = self.hw_to_stim_detectors(raw_syndromes)
        return self.image_mapper.map_to_images(stim_detectors)

    def get_model_input_shape(self, model_type: str) -> tuple:
        if model_type == "graph":
            return (self.graph_mapper.num_nodes, 6)
        if model_type == "image":
            return (self.num_rounds, self.image_mapper.height, self.image_mapper.width)
        raise ValueError(f"Unknown model_type: {model_type}")
