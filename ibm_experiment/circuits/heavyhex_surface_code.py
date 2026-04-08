"""
Heavy-Hex Surface Code Circuit Generator

IBM heavy-hex 토폴로지에 최적화된 surface code 회로를 생성합니다.
Vezvaee et al. "Surface code scaling on heavy-hex superconducting quantum processors"의
fold-unfold SWAP embedding 방식을 구현합니다.

지원 백엔드:
  - Eagle (127q): ibm_yonsei, ibm_strasbourg, ibm_brussels
  - Heron R2 (156q): ibm_fez, ibm_kingston, ibm_marrakesh
  - Heron R3 (156q): ibm_boston, ibm_pittsburgh

Stabilizers (standard rotated surface code d=3):
  X0[D0,D1,D3,D4]  X1[D4,D5,D7,D8]  X2[D3,D6]  X3[D2,D5]
  Z0[D1,D2,D4,D5]  Z1[D3,D4,D6,D7]  Z2[D0,D1]  Z3[D7,D8]

Measurement Protocol (per QEC cycle):
  Half-round 1 (Z stabilizers):
    Z0: fold D2→D1, D5→D4 → measure weight-2 on ancilla
    Z1: fold D4→D3, D7→D6 → measure weight-2 on ancilla
    Z2: boundary D0,D1 on ancilla
    Z3: boundary D7,D8 on ancilla
  Half-round 2 (X stabilizers):
    X0: fold D1→D0, D4→D3 → measure weight-2 on ancilla
    X1: fold D5→D4, D8→D7 → measure weight-2 on ancilla
    X2: boundary D3,D6 on ancilla
    X3: relay measure D2,D5 on ancilla
"""

from qiskit import QuantumCircuit, ClassicalRegister
from typing import List, Dict


# ==============================================================================
# Backend-specific qubit layouts
# ==============================================================================
LAYOUTS = {
    # Eagle 127-qubit backends
    "eagle": {
        "data": {0: 22, 1: 24, 2: 26, 3: 41, 4: 43, 5: 45, 6: 60, 7: 62, 8: 64},
        "z_ancilla": {0: 34, 1: 53, 2: 23, 3: 63},
        "x_ancilla": {0: 34, 1: 54, 2: 53, 3: 35},
        "z_folds": {
            0: [(26, 25, 24), (45, 44, 43)],   # Z0: D2→D1 via 25, D5→D4 via 44
            1: [(43, 42, 41), (62, 61, 60)],    # Z1: D4→D3 via 42, D7→D6 via 61
        },
        "x_folds": {
            0: [(24, 23, 22), (43, 42, 41)],    # X0: D1→D0 via 23, D4→D3 via 42
            1: [(45, 44, 43), (64, 63, 62)],    # X1: D5→D4 via 44, D8→D7 via 63
        },
        "x3_relay_d2": [28, 27],                # anc35 → 28 → 27 → D2(26)
        "x3_relay_d5": [47, 46],                # anc35 → 47 → 46 → D5(45)
        "shared_z_ancilla": [34, 53, 23, 63],
        "backends": ["ibm_yonsei", "ibm_strasbourg", "ibm_brussels"],
    },
    # Heron 156-qubit backends
    "heron": {
        "data": {0: 23, 1: 25, 2: 27, 3: 43, 4: 45, 5: 47, 6: 63, 7: 65, 8: 67},
        "z_ancilla": {0: 37, 1: 56, 2: 24, 3: 66},
        "x_ancilla": {0: 37, 1: 57, 2: 56, 3: 38},
        "z_folds": {
            0: [(27, 26, 25), (47, 46, 45)],    # Z0: D2→D1 via 26, D5→D4 via 46
            1: [(45, 44, 43), (65, 64, 63)],    # Z1: D4→D3 via 44, D7→D6 via 64
        },
        "x_folds": {
            0: [(25, 24, 23), (45, 44, 43)],    # X0: D1→D0 via 24, D4→D3 via 44
            1: [(47, 46, 45), (67, 66, 65)],    # X1: D5→D4 via 46, D8→D7 via 66
        },
        "x3_relay_d2": [29, 28],                # anc38 → 29 → 28 → D2(27)
        "x3_relay_d5": [49, 48],                # anc38 → 49 → 48 → D5(47)
        "shared_z_ancilla": [37, 56, 24, 66],
        "backends": ["ibm_boston", "ibm_pittsburgh", "ibm_fez", "ibm_kingston", "ibm_marrakesh"],
    },
}


def _get_layout(backend_name: str) -> dict:
    """백엔드 이름으로 layout을 찾습니다."""
    for arch, layout in LAYOUTS.items():
        if backend_name in layout["backends"]:
            return layout
    raise ValueError(
        f"Backend '{backend_name}' not supported. "
        f"Supported: {[b for l in LAYOUTS.values() for b in l['backends']]}"
    )


class HeavyHexSurfaceCode:
    """
    Heavy-hex 토폴로지에 최적화된 (3,3) surface code 회로 생성기.
    Fold-unfold 방식으로 SWAP 없이 모든 stabilizer를 측정합니다.
    """

    def __init__(self, distance: int = 3, num_cycles: int = 3,
                 num_rounds: int = None, backend_name: str = "ibm_boston"):
        if distance != 3:
            raise NotImplementedError("Only d=3 supported for heavy-hex embedding.")
        if num_rounds is not None:
            num_cycles = num_rounds

        self.distance = distance
        self.num_cycles = num_cycles
        self.backend_name = backend_name

        # Layout 로드
        self.layout = _get_layout(backend_name)
        self.data_map = self.layout["data"]
        self.data_qubits = [self.data_map[i] for i in range(9)]

        # Stabilizer definitions
        self.x_stab_data = [[0, 1, 3, 4], [4, 5, 7, 8], [3, 6], [2, 5]]
        self.z_stab_data = [[1, 2, 4, 5], [3, 4, 6, 7], [0, 1], [7, 8]]
        self.num_stabilizers = 8
        self.num_data = 9
        self.num_rounds = num_cycles

        # All physical qubits
        self._all_qubits = sorted(set(
            list(self.data_map.values()) +
            list(self.layout["z_ancilla"].values()) +
            list(self.layout["x_ancilla"].values()) +
            [b for folds in self.layout["z_folds"].values() for (_, b, _) in folds] +
            [b for folds in self.layout["x_folds"].values() for (_, b, _) in folds] +
            self.layout["x3_relay_d2"] + self.layout["x3_relay_d5"]
        ))

    # ==========================================================================
    # Low-level gate helpers
    # ==========================================================================

    def _bridge_cx(self, qc: QuantumCircuit, ctrl: int, tgt: int, bridge: int):
        """Bridge를 통한 CX. 비용: 3 CX."""
        qc.cx(ctrl, bridge)
        qc.cx(bridge, tgt)
        qc.cx(ctrl, bridge)

    def _relay_cx(self, qc: QuantumCircuit, ctrl: int, tgt: int, relays: List[int]):
        """다중 relay를 통한 CX."""
        chain = [ctrl] + relays + [tgt]
        for i in range(len(chain) - 1):
            qc.cx(chain[i], chain[i + 1])
        for i in range(len(chain) - 2, 0, -1):
            qc.cx(chain[i - 1], chain[i])

    # ==========================================================================
    # Z half-round
    # ==========================================================================

    def _z_half_round(self, qc, cr_syn, syn_offset):
        L = self.layout
        D = self.data_map
        z_anc = L["z_ancilla"]
        z_folds = L["z_folds"]

        # Z0: fold D2→D1, D5→D4, measure on z_anc[0]
        for src, bridge, tgt in z_folds[0]:
            self._bridge_cx(qc, src, tgt, bridge)
        qc.cx(D[1], z_anc[0])
        qc.cx(D[4], z_anc[0])
        qc.measure(z_anc[0], cr_syn[syn_offset + 0])
        for src, bridge, tgt in z_folds[0]:
            self._bridge_cx(qc, src, tgt, bridge)

        # Z1: fold D4→D3, D7→D6, measure on z_anc[1]
        for src, bridge, tgt in z_folds[1]:
            self._bridge_cx(qc, src, tgt, bridge)
        qc.cx(D[3], z_anc[1])
        qc.cx(D[6], z_anc[1])
        qc.measure(z_anc[1], cr_syn[syn_offset + 1])
        for src, bridge, tgt in z_folds[1]:
            self._bridge_cx(qc, src, tgt, bridge)

        # Z2: boundary D0,D1 on z_anc[2]
        qc.cx(D[0], z_anc[2])
        qc.cx(D[1], z_anc[2])
        qc.measure(z_anc[2], cr_syn[syn_offset + 2])

        # Z3: boundary D7,D8 on z_anc[3]
        qc.cx(D[7], z_anc[3])
        qc.cx(D[8], z_anc[3])
        qc.measure(z_anc[3], cr_syn[syn_offset + 3])

    # ==========================================================================
    # X half-round
    # ==========================================================================

    def _x_half_round(self, qc, cr_syn, syn_offset):
        L = self.layout
        D = self.data_map
        x_anc = L["x_ancilla"]
        x_folds = L["x_folds"]

        # Reset shared ancilla (Z half-round에서 사용된 것들)
        for a in L["shared_z_ancilla"]:
            qc.reset(a)

        # X0: fold D1→D0, D4→D3, measure on x_anc[0]
        for src, bridge, tgt in x_folds[0]:
            self._bridge_cx(qc, src, tgt, bridge)
        qc.h(x_anc[0])
        qc.cx(x_anc[0], D[1])
        qc.cx(x_anc[0], D[4])
        qc.h(x_anc[0])
        qc.measure(x_anc[0], cr_syn[syn_offset + 0])
        for src, bridge, tgt in x_folds[0]:
            self._bridge_cx(qc, src, tgt, bridge)

        # X1: fold D5→D4, D8→D7, measure on x_anc[1]
        for src, bridge, tgt in x_folds[1]:
            self._bridge_cx(qc, src, tgt, bridge)
        qc.h(x_anc[1])
        qc.cx(x_anc[1], D[5])
        qc.cx(x_anc[1], D[8])
        qc.h(x_anc[1])
        qc.measure(x_anc[1], cr_syn[syn_offset + 1])
        for src, bridge, tgt in x_folds[1]:
            self._bridge_cx(qc, src, tgt, bridge)

        # X2: boundary D3,D6 on x_anc[2]
        qc.h(x_anc[2])
        qc.cx(x_anc[2], D[3])
        qc.cx(x_anc[2], D[6])
        qc.h(x_anc[2])
        qc.measure(x_anc[2], cr_syn[syn_offset + 2])

        # X3: relay D2,D5 on x_anc[3]
        qc.h(x_anc[3])
        self._relay_cx(qc, x_anc[3], D[2], L["x3_relay_d2"])
        self._relay_cx(qc, x_anc[3], D[5], L["x3_relay_d5"])
        qc.h(x_anc[3])
        qc.measure(x_anc[3], cr_syn[syn_offset + 3])

    # ==========================================================================
    # Circuit construction
    # ==========================================================================

    def build_circuit(self, initial_state: int = 0) -> QuantumCircuit:
        """Heavy-hex 최적화 surface code memory 회로를 생성합니다."""
        n_qubits = max(self._all_qubits) + 1

        cr_syn = ClassicalRegister(self.num_stabilizers * self.num_cycles, 'syn')
        cr_data = ClassicalRegister(self.num_data, 'data')
        qc = QuantumCircuit(n_qubits)
        qc.add_register(cr_syn)
        qc.add_register(cr_data)

        # Reset all used qubits
        for q in self._all_qubits:
            qc.reset(q)

        # Logical initial state
        if initial_state == 1:
            for i in [0, 1, 2]:
                qc.x(self.data_map[i])

        # QEC cycles
        for cycle in range(self.num_cycles):
            so = cycle * self.num_stabilizers
            qc.barrier()
            self._z_half_round(qc, cr_syn, so)
            qc.barrier()
            self._x_half_round(qc, cr_syn, so + 4)

        # Final data measurement
        qc.barrier()
        for i in range(9):
            qc.measure(self.data_map[i], cr_data[i])

        return qc

    # ==========================================================================
    # Pipeline interface
    # ==========================================================================

    def get_syndrome_indices(self) -> dict:
        """실험 파이프라인과 호환되는 syndrome 정보."""
        return {
            "num_data": self.num_data,
            "num_stabilizers": self.num_stabilizers,
            "num_rounds": self.num_cycles,
            "x_stabilizers": self.x_stab_data,
            "z_stabilizers": self.z_stab_data,
            "logical_x": [0, 1, 2],
            "logical_z": [0, 3, 6],
        }

    def get_circuit_summary(self) -> str:
        return (
            f"=== Heavy-Hex Surface Code ===\n"
            f"Distance: {self.distance}\n"
            f"QEC Cycles: {self.num_cycles}\n"
            f"Backend: {self.backend_name}\n"
            f"Data qubits: {self.data_qubits}\n"
            f"Physical qubits: {len(self._all_qubits)}"
        )

    def get_transpile_options(self) -> dict:
        return {"optimization_level": 1}


# ==============================================================================
# 단독 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    from qiskit_aer import AerSimulator

    for bname in ["ibm_boston", "ibm_yonsei"]:
        print(f"\n{'='*50}")
        print(f"  Testing: {bname}")
        print(f"{'='*50}")

        try:
            sc = HeavyHexSurfaceCode(distance=3, num_cycles=3, backend_name=bname)
        except ValueError as e:
            print(f"  Skipped: {e}")
            continue

        print(sc.get_circuit_summary())
        qc = sc.build_circuit(initial_state=0)

        ops = qc.count_ops()
        print(f"\nCircuit: depth={qc.depth()}, CX={ops.get('cx',0)}")

        # AerSimulator verification
        n_qubits = max(sc._all_qubits) + 1
        sim = AerSimulator(n_qubits=n_qubits)
        result = sim.run(qc, shots=1000).result()
        counts = result.get_counts()

        correct = 0
        total = sum(counts.values())
        for bitstr, count in counts.items():
            parts = bitstr.split()
            data_bits = parts[0] if len(parts) >= 2 else bitstr[-9:]
            d0 = int(data_bits[-1])
            d3 = int(data_bits[-4])
            d6 = int(data_bits[-7])
            if (d0 + d3 + d6) % 2 == 0:
                correct += count

        print(f"Logical Z = 0: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"{'✅ PASS' if correct == total else '❌ FAIL'}")