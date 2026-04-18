"""
Rotated Surface Code 회로 생성기 (IBM Eagle용)

d=3: 9 data + 8 ancilla/round = 최소 17 qubits (127-qubit Eagle에 충분)
d=5: 25 data + 24 ancilla/round = 최소 49 qubits
d=7: 49 data + 48 ancilla/round = 최소 97 qubits

Stabilizer 구조 (d=3 예시):
  Data qubits (3x3 grid):
    D0  D1  D2
    D3  D4  D5
    D6  D7  D8

  X stabilizers: {0,1,3,4}, {4,5,7,8}, {3,6}, {2,5}
  Z stabilizers: {1,2,4,5}, {3,4,6,7}, {0,1}, {7,8}

  Logical X: X on top row (q0, q1, q2)
  Logical Z: Z on left column (q0, q3, q6)
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np


class SurfaceCodeCircuit:
    def __init__(self, distance: int, num_rounds: int = 1,
                 physical_qubits: dict = None):
        """
        Rotated surface code QEC memory circuit.

        Args:
            distance: code distance (3, 5, 7)
            num_rounds: number of QEC rounds
            physical_qubits: optional dict {logical_idx -> physical_idx}.
                Keys 0..d^2-1 map data qubits; keys d^2..d^2+num_stab-1 map
                ancilla (X stabs then Z stabs, matching x_stabilizers +
                z_stabilizers). When provided, the circuit uses a single
                ancilla register with reset between rounds (ancilla reuse),
                so the physical layout only needs d^2 + num_stab qubits
                rather than d^2 + num_stab * num_rounds.
                When None, legacy per-round ancilla allocation is used.
        """
        self.distance = distance
        self.num_rounds = num_rounds
        self.physical_qubits = physical_qubits
        self.reuse_ancilla = physical_qubits is not None

        self.num_data = distance ** 2
        self.x_stabilizers, self.z_stabilizers = self._generate_stabilizers()
        self.num_stabilizers = len(self.x_stabilizers) + len(self.z_stabilizers)

        # Logical operators
        self.logical_x = list(range(distance))  # top row
        self.logical_z = list(range(0, distance ** 2, distance))  # left column

        if physical_qubits is not None:
            expected = self.num_data + self.num_stabilizers
            if len(physical_qubits) != expected:
                raise ValueError(
                    f"physical_qubits must have {expected} entries "
                    f"({self.num_data} data + {self.num_stabilizers} ancilla), "
                    f"got {len(physical_qubits)}."
                )

    def _generate_stabilizers(self):
        """Distance에 따른 stabilizer를 자동 생성합니다."""
        d = self.distance
        x_stabs = []
        z_stabs = []

        # Bulk stabilizers (weight-4)
        for r in range(d - 1):
            for c in range(d - 1):
                qubits = [r * d + c, r * d + c + 1, (r + 1) * d + c, (r + 1) * d + c + 1]
                if (r + c) % 2 == 0:
                    x_stabs.append(qubits)
                else:
                    z_stabs.append(qubits)

        # Left boundary: X stabilizers
        for r in range(d - 1):
            if (r + (-1)) % 2 == 0:
                x_stabs.append([r * d, (r + 1) * d])

        # Right boundary: X stabilizers
        for r in range(d - 1):
            if (r + (d - 1)) % 2 == 0:
                x_stabs.append([r * d + (d - 1), (r + 1) * d + (d - 1)])

        # Top boundary: Z stabilizers
        for c in range(d - 1):
            if ((-1) + c) % 2 == 1:
                z_stabs.append([c, c + 1])

        # Bottom boundary: Z stabilizers
        for c in range(d - 1):
            if ((d - 1) + c) % 2 == 1:
                z_stabs.append([(d - 1) * d + c, (d - 1) * d + c + 1])

        return x_stabs, z_stabs

    def build_circuit(self, initial_state: int = 0) -> QuantumCircuit:
        """
        Surface Code QEC 회로를 생성합니다.

        Args:
            initial_state: 0이면 |0⟩_L, 1이면 |1⟩_L

        Returns:
            Qiskit QuantumCircuit
        """
        if self.reuse_ancilla:
            return self._build_reuse_circuit(initial_state)
        return self._build_legacy_circuit(initial_state)

    def _build_legacy_circuit(self, initial_state: int) -> QuantumCircuit:
        num_stab = self.num_stabilizers

        data = QuantumRegister(self.num_data, "data")
        ancilla_regs = []
        syndrome_cregs = []

        for r in range(self.num_rounds):
            anc = QuantumRegister(num_stab, f"anc_r{r}")
            syn = ClassicalRegister(num_stab, f"syn_r{r}")
            ancilla_regs.append(anc)
            syndrome_cregs.append(syn)

        data_creg = ClassicalRegister(self.num_data, "data_meas")

        qc = QuantumCircuit(data, *ancilla_regs, *syndrome_cregs, data_creg)

        if initial_state == 1:
            for q in self.logical_x:
                qc.x(data[q])

        qc.barrier()

        for r in range(self.num_rounds):
            anc = ancilla_regs[r]
            syn = syndrome_cregs[r]
            anc_idx = 0

            for stab_qubits in self.x_stabilizers:
                qc.h(anc[anc_idx])
                for dq in stab_qubits:
                    qc.cx(anc[anc_idx], data[dq])
                qc.h(anc[anc_idx])
                qc.measure(anc[anc_idx], syn[anc_idx])
                anc_idx += 1

            for stab_qubits in self.z_stabilizers:
                for dq in stab_qubits:
                    qc.cx(data[dq], anc[anc_idx])
                qc.measure(anc[anc_idx], syn[anc_idx])
                anc_idx += 1

            qc.barrier()

        for i in range(self.num_data):
            qc.measure(data[i], data_creg[i])

        return qc

    def _build_reuse_circuit(self, initial_state: int) -> QuantumCircuit:
        """Single ancilla register, reset between rounds for hardware mapping."""
        num_stab = self.num_stabilizers

        data = QuantumRegister(self.num_data, "data")
        anc = QuantumRegister(num_stab, "anc")
        syndrome_cregs = [
            ClassicalRegister(num_stab, f"syn_r{r}")
            for r in range(self.num_rounds)
        ]
        data_creg = ClassicalRegister(self.num_data, "data_meas")

        qc = QuantumCircuit(data, anc, *syndrome_cregs, data_creg)

        if initial_state == 1:
            for q in self.logical_x:
                qc.x(data[q])

        qc.barrier()

        for r in range(self.num_rounds):
            syn = syndrome_cregs[r]

            if r > 0:
                for i in range(num_stab):
                    qc.reset(anc[i])

            anc_idx = 0
            for stab_qubits in self.x_stabilizers:
                qc.h(anc[anc_idx])
                for dq in stab_qubits:
                    qc.cx(anc[anc_idx], data[dq])
                qc.h(anc[anc_idx])
                qc.measure(anc[anc_idx], syn[anc_idx])
                anc_idx += 1

            for stab_qubits in self.z_stabilizers:
                for dq in stab_qubits:
                    qc.cx(data[dq], anc[anc_idx])
                qc.measure(anc[anc_idx], syn[anc_idx])
                anc_idx += 1

            qc.barrier()

        for i in range(self.num_data):
            qc.measure(data[i], data_creg[i])

        return qc

    def get_initial_layout(self) -> list:
        """
        Physical qubit list indexed by virtual-circuit position, for passing
        as `initial_layout=` to `qiskit.transpile`. Requires physical_qubits
        to have been provided at construction.
        """
        if self.physical_qubits is None:
            raise ValueError("physical_qubits was not provided at construction.")
        layout = []
        for i in range(self.num_data):
            layout.append(self.physical_qubits[i])
        for i in range(self.num_stabilizers):
            layout.append(self.physical_qubits[self.num_data + i])
        return layout

    def get_syndrome_indices(self) -> dict:
        """bitstring 파싱을 위한 인덱스 매핑을 반환합니다."""
        return {
            "num_data": self.num_data,
            "num_stabilizers": self.num_stabilizers,
            "num_rounds": self.num_rounds,
            "x_stabilizers": self.x_stabilizers,
            "z_stabilizers": self.z_stabilizers,
            "logical_x": self.logical_x,
            "logical_z": self.logical_z,
            "depth7": False,
        }

    def get_circuit_summary(self) -> str:
        d = self.distance
        num_x = len(self.x_stabilizers)
        num_z = len(self.z_stabilizers)
        total_qubits = self.num_data + self.num_stabilizers * self.num_rounds
        syn_bits = self.num_stabilizers * self.num_rounds
        total_cbits = syn_bits + self.num_data

        return (
            f"=== Surface Code Circuit Summary ===\n"
            f"Distance: {d}\n"
            f"Data Qubits: {self.num_data}\n"
            f"Stabilizers: {self.num_stabilizers} ({num_x} X-type + {num_z} Z-type)\n"
            f"QEC Rounds: {self.num_rounds}\n"
            f"Ancilla per Round: {self.num_stabilizers}\n"
            f"Total Qubits: {total_qubits}\n"
            f"Total Classical Bits: {total_cbits}\n"
            f"  - Syndrome Bits: {syn_bits}\n"
            f"  - Data Meas Bits: {self.num_data}\n"
            f"Logical X: X on qubits {self.logical_x}\n"
            f"Logical Z: Z on qubits {self.logical_z}"
        )


if __name__ == "__main__":
    for d in [3, 5, 7]:
        sc = SurfaceCodeCircuit(distance=d, num_rounds=d)
        print(sc.get_circuit_summary())
        print(f"X stabs: {sc.x_stabilizers}")
        print(f"Z stabs: {sc.z_stabilizers}")
        print()
