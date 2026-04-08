"""
Qiskit 기반 2D Triangular Color Code 회로 생성기

d=3: [[7,1,3]] Steane Code (하드코딩, 검증 완료)
d=5: [[19,1,5]] Color Code (ancilla reuse, mid-circuit measurement)

회로 구성:
  d=3: 인코딩(systematic generator) + QEC 라운드 + 최종 측정
  d=5: stabilizer projection 인코딩 + QEC 라운드 + 최종 측정
       ancilla reuse: X용 1개 + Z용 1개 = 2 ancilla → 총 21큐빗
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ==============================================================================
# d=3 [[7,1,3]] Steane Code 정의
# ==============================================================================
# 
# Parity Check Matrix H (= C⊥ generator):
#   H = [[1,1,1,1,0,0,0],
#        [0,1,1,0,1,1,0],
#        [0,0,1,1,0,1,1]]
#
# Stabilizer Plaquettes (X-type & Z-type 동일 support):
#   Plaquette R (Red):   qubits {0, 1, 2, 3}
#   Plaquette G (Green): qubits {1, 2, 4, 5}
#   Plaquette B (Blue):  qubits {2, 3, 5, 6}
#
# Logical Operators:
#   Z_L = Z₀Z₁Z₄  (weight-3 codeword of C)
#   X_L = X₀X₂X₅  (weight-3 codeword of C, anticommutes with Z_L)
#
# Encoding |0⟩_L (Reduced Systematic Generator of C⊥):
#   c₀ = m₀, c₁ = m₁, c₂ = m₂
#   c₃ = m₀⊕m₁⊕m₂, c₄ = m₀⊕m₁, c₅ = m₀⊕m₂, c₆ = m₁⊕m₂
# ==============================================================================

STEANE_CODE = {
    "num_data": 7,
    "stabilizers": [
        # (color, type, qubit_indices)
        # X-type stabilizers
        {"color": "R", "type": "X", "qubits": [0, 1, 2, 3]},
        {"color": "G", "type": "X", "qubits": [1, 2, 4, 5]},
        {"color": "B", "type": "X", "qubits": [2, 3, 5, 6]},
        # Z-type stabilizers
        {"color": "R", "type": "Z", "qubits": [0, 1, 2, 3]},
        {"color": "G", "type": "Z", "qubits": [1, 2, 4, 5]},
        {"color": "B", "type": "Z", "qubits": [2, 3, 5, 6]},
    ],
    "logical_z": [0, 1, 4],  # Z_L = Z₀Z₁Z₄
    "logical_x": [0, 2, 5],  # X_L = X₀X₂X₅
}


# ==============================================================================
# d=5 [[19,1,5]] Color Code 정의 (Stim에서 추출)
# ==============================================================================
#
# Plaquette 구조 (X/Z stabilizer 동일 support):
#   P0 (R): [1,2,5,6]          P1 (R): [3,4,7,8]
#   P2 (G): [0,1,5,9]          P3 (G): [2,3,6,7,10,11]
#   P4 (B): [5,6,9,10,12,13]   P5 (B): [7,8,11,14]
#   P6 (R): [10,11,13,14,15,16] P7 (G): [12,13,15,17]
#   P8 (B): [15,16,17,18]
#
# 색상 분류: R={P0,P1,P6}, G={P2,P3,P7}, B={P4,P5,P8}
#
# Logical Operators (weight-5, self-dual):
#   Z_L = Z₀Z₁Z₂Z₃Z₄
#   X_L = X₀X₁X₂X₃X₄
#
# 인코딩: Stabilizer projection
#   |0⟩^19 → 이미 Z_L=+1, Z-stab +1 eigenstate
#   첫 라운드 X-stab 측정이 code space로 projection
# ==============================================================================

D5_COLOR_CODE = {
    "num_data": 19,
    "plaquettes": [
        {"index": 0, "color": "R", "qubits": [1, 2, 5, 6]},
        {"index": 1, "color": "R", "qubits": [3, 4, 7, 8]},
        {"index": 2, "color": "G", "qubits": [0, 1, 5, 9]},
        {"index": 3, "color": "G", "qubits": [2, 3, 6, 7, 10, 11]},
        {"index": 4, "color": "B", "qubits": [5, 6, 9, 10, 12, 13]},
        {"index": 5, "color": "B", "qubits": [7, 8, 11, 14]},
        {"index": 6, "color": "R", "qubits": [10, 11, 13, 14, 15, 16]},
        {"index": 7, "color": "G", "qubits": [12, 13, 15, 17]},
        {"index": 8, "color": "B", "qubits": [15, 16, 17, 18]},
    ],
    "stabilizers": [
        # X-type stabilizers (P0-P8)
        {"color": "R", "type": "X", "qubits": [1, 2, 5, 6]},
        {"color": "R", "type": "X", "qubits": [3, 4, 7, 8]},
        {"color": "G", "type": "X", "qubits": [0, 1, 5, 9]},
        {"color": "G", "type": "X", "qubits": [2, 3, 6, 7, 10, 11]},
        {"color": "B", "type": "X", "qubits": [5, 6, 9, 10, 12, 13]},
        {"color": "B", "type": "X", "qubits": [7, 8, 11, 14]},
        {"color": "R", "type": "X", "qubits": [10, 11, 13, 14, 15, 16]},
        {"color": "G", "type": "X", "qubits": [12, 13, 15, 17]},
        {"color": "B", "type": "X", "qubits": [15, 16, 17, 18]},
        # Z-type stabilizers (P0-P8)
        {"color": "R", "type": "Z", "qubits": [1, 2, 5, 6]},
        {"color": "R", "type": "Z", "qubits": [3, 4, 7, 8]},
        {"color": "G", "type": "Z", "qubits": [0, 1, 5, 9]},
        {"color": "G", "type": "Z", "qubits": [2, 3, 6, 7, 10, 11]},
        {"color": "B", "type": "Z", "qubits": [5, 6, 9, 10, 12, 13]},
        {"color": "B", "type": "Z", "qubits": [7, 8, 11, 14]},
        {"color": "R", "type": "Z", "qubits": [10, 11, 13, 14, 15, 16]},
        {"color": "G", "type": "Z", "qubits": [12, 13, 15, 17]},
        {"color": "B", "type": "Z", "qubits": [15, 16, 17, 18]},
    ],
    "logical_z": [0, 1, 2, 3, 4],  # Z_L = Z₀Z₁Z₂Z₃Z₄
    "logical_x": [0, 1, 2, 3, 4],  # X_L = X₀X₁X₂X₃X₄ (self-dual)
}


class ColorCodeCircuit:
    """
    2D Triangular Color Code의 Qiskit 회로를 생성합니다.
    
    Parameters:
        distance (int): Code distance (3, 5, 7, ...)
        num_rounds (int): QEC 신드롬 측정 라운드 수
    """
    
    def __init__(self, distance: int = 3, num_rounds: int = 1):
        self.distance = distance
        self.num_rounds = num_rounds

        # 코드 구조 로드
        self.code_def = self._load_code_definition()

        # 큐빗 레이아웃 계산
        self.num_data = self.code_def["num_data"]
        self.stabilizers = self.code_def["stabilizers"]
        self.num_stabilizers = len(self.stabilizers)
        self.logical_z = self.code_def["logical_z"]
        self.logical_x = self.code_def["logical_x"]

        # Ancilla reuse 여부: d>=5에서 mid-circuit measurement + reset 사용
        self.ancilla_reuse = (distance >= 5)

        if self.ancilla_reuse:
            # d=5: X용 1개 + Z용 1개 = 2 ancilla (reuse via reset)
            self.num_ancilla = 2
            self.total_qubits = self.num_data + self.num_ancilla
        else:
            # d=3: 라운드별로 별도 ancilla 할당
            self.num_ancilla_per_round = self.num_stabilizers
            self.total_ancilla = self.num_ancilla_per_round * self.num_rounds
            self.total_qubits = self.num_data + self.total_ancilla

        # Classical register 크기:
        #   - 라운드별 신드롬 비트: num_stabilizers * num_rounds
        #   - 데이터 큐빗 최종 측정: num_data
        self.num_syndrome_bits = self.num_stabilizers * self.num_rounds
        self.num_data_bits = self.num_data
        
    def _load_code_definition(self) -> dict:
        """Distance에 따른 코드 정의를 로드합니다."""
        if self.distance == 3:
            return STEANE_CODE
        elif self.distance == 5:
            return D5_COLOR_CODE
        else:
            raise ValueError(f"Distance {self.distance} is not supported. Use 3 or 5.")
    
    def build_circuit(self, initial_state: int = 0) -> QuantumCircuit:
        """
        전체 QEC 회로를 생성합니다.

        Args:
            initial_state: 논리적 초기 상태 (0 = |0⟩_L, 1 = |1⟩_L)

        Returns:
            QuantumCircuit: 인코딩 + QEC 라운드 + 최종 측정이 포함된 회로
        """
        if self.ancilla_reuse:
            return self._build_circuit_ancilla_reuse(initial_state)
        else:
            return self._build_circuit_separate_ancilla(initial_state)

    def _build_circuit_separate_ancilla(self, initial_state: int) -> QuantumCircuit:
        """d=3: 라운드별 별도 ancilla 할당 (기존 방식)."""
        data = QuantumRegister(self.num_data, name="data")
        ancilla_regs = []
        for r in range(self.num_rounds):
            anc = QuantumRegister(self.num_stabilizers, name=f"anc_r{r}")
            ancilla_regs.append(anc)

        syndrome_regs = []
        for r in range(self.num_rounds):
            syn = ClassicalRegister(self.num_stabilizers, name=f"syn_r{r}")
            syndrome_regs.append(syn)
        data_meas = ClassicalRegister(self.num_data, name="data_meas")

        qc = QuantumCircuit(data, *ancilla_regs, *syndrome_regs, data_meas)

        qc.barrier(label="Encoding")
        self._encode_logical_state(qc, data, initial_state)

        for r in range(self.num_rounds):
            qc.barrier(label=f"QEC Round {r+1}")
            self._measure_stabilizers(qc, data, ancilla_regs[r], syndrome_regs[r])

        qc.barrier(label="Final Measurement")
        qc.measure(data, data_meas)
        return qc

    def _build_circuit_ancilla_reuse(self, initial_state: int) -> QuantumCircuit:
        """
        d=5: Ancilla reuse 방식 (mid-circuit measurement + reset).

        X용 ancilla 1개 + Z용 ancilla 1개 = 총 2 ancilla.
        각 stabilizer를 순차적으로 측정 후 reset & reuse.
        """
        data = QuantumRegister(self.num_data, name="data")
        anc = QuantumRegister(2, name="anc")  # anc[0]=X용, anc[1]=Z용

        syndrome_regs = []
        for r in range(self.num_rounds):
            syn = ClassicalRegister(self.num_stabilizers, name=f"syn_r{r}")
            syndrome_regs.append(syn)
        data_meas = ClassicalRegister(self.num_data, name="data_meas")

        qc = QuantumCircuit(data, anc, *syndrome_regs, data_meas)

        # Step 1: 인코딩 (stabilizer projection 방식)
        qc.barrier(label="Encoding")
        self._encode_logical_state(qc, data, initial_state)

        # Step 2: QEC 라운드
        for r in range(self.num_rounds):
            qc.barrier(label=f"QEC Round {r+1}")
            self._measure_stabilizers_reuse(qc, data, anc, syndrome_regs[r])

        # Step 3: 데이터 큐빗 최종 측정
        qc.barrier(label="Final Measurement")
        qc.measure(data, data_meas)
        return qc
    
    def _encode_logical_state(self, qc: QuantumCircuit, data: QuantumRegister, state: int):
        """
        논리적 |0⟩_L 또는 |1⟩_L을 인코딩합니다.

        d=3: Systematic generator를 이용한 직접 인코딩
        d=5: Stabilizer projection 방식
             |0⟩^19은 이미 Z_L=+1, Z-stab +1 eigenstate.
             첫 라운드 X-stab 측정이 code space로 projection.
        """
        if self.distance == 3:
            self._encode_d3(qc, data, state)
        elif self.distance == 5:
            self._encode_d5(qc, data, state)

    def _encode_d3(self, qc: QuantumCircuit, data: QuantumRegister, state: int):
        """d=3 인코딩: Reduced Systematic Generator of C⊥."""
        # 메시지 비트 중첩
        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])

        # 패리티 비트 계산
        # q3 = q0 ⊕ q1 ⊕ q2
        qc.cx(data[0], data[3])
        qc.cx(data[1], data[3])
        qc.cx(data[2], data[3])

        # q4 = q0 ⊕ q1
        qc.cx(data[0], data[4])
        qc.cx(data[1], data[4])

        # q5 = q0 ⊕ q2
        qc.cx(data[0], data[5])
        qc.cx(data[2], data[5])

        # q6 = q1 ⊕ q2
        qc.cx(data[1], data[6])
        qc.cx(data[2], data[6])

        # |1⟩_L = X_L |0⟩_L
        if state == 1:
            for q in self.logical_x:
                qc.x(data[q])

    def _encode_d5(self, qc: QuantumCircuit, data: QuantumRegister, state: int):
        """
        d=5 인코딩: Stabilizer projection 방식.

        |0⟩^19 → 모든 Z-stab +1, Z_L = +1.
        |1⟩_L이 필요하면 X_L 적용 (projection 전에).
        첫 QEC 라운드의 X-stab 측정이 code space로 projection.
        """
        if state == 1:
            for q in self.logical_x:
                qc.x(data[q])
    
    def _measure_stabilizers(self, qc: QuantumCircuit, data: QuantumRegister,
                             ancilla: QuantumRegister, syndrome: ClassicalRegister):
        """
        1라운드의 Stabilizer 측정 회로를 구성합니다.
        
        X-type: ancilla를 |+⟩로 준비 → CNOT(ancilla→data) → H → 측정
        Z-type: ancilla를 |0⟩로 준비 → CNOT(data→ancilla) → 측정
        """
        for i, stab in enumerate(self.stabilizers):
            if stab["type"] == "X":
                # X stabilizer 측정
                qc.h(ancilla[i])  # |0⟩ → |+⟩
                for q in stab["qubits"]:
                    qc.cx(ancilla[i], data[q])  # ancilla=control, data=target
                qc.h(ancilla[i])  # X-basis 측정을 위한 역변환
                qc.measure(ancilla[i], syndrome[i])
                
            elif stab["type"] == "Z":
                # Z stabilizer 측정 (ancilla는 이미 |0⟩)
                for q in stab["qubits"]:
                    qc.cx(data[q], ancilla[i])  # data=control, ancilla=target
                qc.measure(ancilla[i], syndrome[i])
    
    def _measure_stabilizers_reuse(self, qc: QuantumCircuit, data: QuantumRegister,
                                    anc: QuantumRegister, syndrome: ClassicalRegister):
        """
        1라운드의 Stabilizer 측정 (ancilla reuse 방식).

        anc[0]: X-stabilizer 측정용 (순차 reuse)
        anc[1]: Z-stabilizer 측정용 (순차 reuse)

        각 stabilizer 측정 후 reset으로 ancilla를 |0⟩으로 되돌림.
        """
        for i, stab in enumerate(self.stabilizers):
            if stab["type"] == "X":
                qc.reset(anc[0])
                qc.h(anc[0])
                for q in stab["qubits"]:
                    qc.cx(anc[0], data[q])
                qc.h(anc[0])
                qc.measure(anc[0], syndrome[i])

            elif stab["type"] == "Z":
                qc.reset(anc[1])
                for q in stab["qubits"]:
                    qc.cx(data[q], anc[1])
                qc.measure(anc[1], syndrome[i])

    def get_syndrome_indices(self) -> dict:
        """
        Classical register에서 신드롬 비트의 위치 매핑을 반환합니다.
        
        Returns:
            dict: {
                "syndrome_ranges": [(start, end), ...],  # 라운드별 신드롬 비트 범위
                "data_range": (start, end),               # 데이터 큐빗 비트 범위
                "stabilizer_info": [...]                   # 각 stabilizer의 메타정보
            }
        """
        total_classical = self.num_syndrome_bits + self.num_data_bits
        
        syndrome_ranges = []
        offset = 0
        for r in range(self.num_rounds):
            start = offset
            end = offset + self.num_stabilizers
            syndrome_ranges.append((start, end))
            offset = end
        
        data_range = (offset, offset + self.num_data)
        
        return {
            "syndrome_ranges": syndrome_ranges,
            "data_range": data_range,
            "stabilizer_info": self.stabilizers,
            "logical_z": self.logical_z,
            "logical_x": self.logical_x,
            "num_data": self.num_data,
            "num_stabilizers": self.num_stabilizers,
            "num_rounds": self.num_rounds,
        }
    
    def get_circuit_summary(self) -> str:
        """회로의 요약 정보를 반환합니다."""
        if self.ancilla_reuse:
            ancilla_line = f"Ancilla: {self.num_ancilla} (reuse via mid-circuit reset)\n"
        else:
            ancilla_line = f"Ancilla per Round: {self.num_ancilla_per_round}\n"
        return (
            f"=== Color Code Circuit Summary ===\n"
            f"Distance: {self.distance}\n"
            f"Data Qubits: {self.num_data}\n"
            f"Stabilizers: {self.num_stabilizers} ({self.num_stabilizers // 2} X-type + {self.num_stabilizers // 2} Z-type)\n"
            f"QEC Rounds: {self.num_rounds}\n"
            f"{ancilla_line}"
            f"Total Qubits: {self.total_qubits}\n"
            f"Total Classical Bits: {self.num_syndrome_bits + self.num_data_bits}\n"
            f"  - Syndrome Bits: {self.num_syndrome_bits}\n"
            f"  - Data Meas Bits: {self.num_data_bits}\n"
            f"Logical Z: Z on qubits {self.logical_z}\n"
            f"Logical X: X on qubits {self.logical_x}\n"
        )


# ==============================================================================
# 단독 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    # d=3 회로 생성 테스트
    print("=" * 60)
    cc3 = ColorCodeCircuit(distance=3, num_rounds=1)
    print(cc3.get_circuit_summary())

    qc3 = cc3.build_circuit(initial_state=0)
    print(f"Circuit depth: {qc3.depth()}")
    print(f"Gate counts: {dict(qc3.count_ops())}")
    print(f"\n{qc3.draw(output='text', fold=120)}")

    indices3 = cc3.get_syndrome_indices()
    print(f"\nSyndrome indices: {indices3}")

    # d=5 회로 생성 테스트
    print("\n" + "=" * 60)
    cc5 = ColorCodeCircuit(distance=5, num_rounds=1)
    print(cc5.get_circuit_summary())

    qc5 = cc5.build_circuit(initial_state=0)
    print(f"Circuit depth: {qc5.depth()}")
    print(f"Gate counts: {dict(qc5.count_ops())}")
    print(f"Num qubits: {qc5.num_qubits}")

    indices5 = cc5.get_syndrome_indices()
    print(f"\nSyndrome indices: {indices5}")

    # d=5, 2라운드 테스트
    print("\n" + "=" * 60)
    cc5r2 = ColorCodeCircuit(distance=5, num_rounds=2)
    print(cc5r2.get_circuit_summary())
    qc5r2 = cc5r2.build_circuit(initial_state=0)
    print(f"Circuit depth: {qc5r2.depth()}")
    print(f"Num qubits: {qc5r2.num_qubits}")
    print(f"Num classical bits: {qc5r2.num_clbits}")
