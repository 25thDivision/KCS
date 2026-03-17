"""
Qiskit 기반 2D Triangular Color Code 회로 생성기

d=3: [[7,1,3]] Steane Code (하드코딩, 검증 완료)
d=5, d=7: 확장 가능한 프레임워크 (Tempo 대응)

회로 구성:
  1. 논리적 |0⟩_L 인코딩
  2. QEC 라운드 (Stabilizer 측정)
  3. 데이터 큐빗 최종 측정
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
        
        # 앵실러: 라운드별로 별도 할당 (mid-circuit measurement 없이)
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
            return self._generate_code_definition(5)
        elif self.distance == 7:
            return self._generate_code_definition(7)
        else:
            raise ValueError(f"Distance {self.distance} is not supported. Use 3, 5, or 7.")
    
    def _generate_code_definition(self, d: int) -> dict:
        """
        d >= 5에 대한 Color Code 정의를 좌표 기반으로 생성합니다.
        
        TODO: Tempo (100 큐빗) 출시 시 구현 예정
        현재는 placeholder로, d=3과 동일한 인터페이스를 제공합니다.
        """
        raise NotImplementedError(
            f"Distance {d} color code is not yet implemented. "
            f"Currently only d=3 is supported. "
            f"d=5 support will be added when IonQ Tempo becomes available."
        )
    
    def build_circuit(self, initial_state: int = 0) -> QuantumCircuit:
        """
        전체 QEC 회로를 생성합니다.
        
        Args:
            initial_state: 논리적 초기 상태 (0 = |0⟩_L, 1 = |1⟩_L)
        
        Returns:
            QuantumCircuit: 인코딩 + QEC 라운드 + 최종 측정이 포함된 회로
        """
        # Quantum Registers
        data = QuantumRegister(self.num_data, name="data")
        ancilla_regs = []
        for r in range(self.num_rounds):
            anc = QuantumRegister(self.num_stabilizers, name=f"anc_r{r}")
            ancilla_regs.append(anc)
        
        # Classical Registers
        syndrome_regs = []
        for r in range(self.num_rounds):
            syn = ClassicalRegister(self.num_stabilizers, name=f"syn_r{r}")
            syndrome_regs.append(syn)
        data_meas = ClassicalRegister(self.num_data, name="data_meas")
        
        # Circuit 생성
        qc = QuantumCircuit(data, *ancilla_regs, *syndrome_regs, data_meas)
        
        # Step 1: 논리적 상태 인코딩
        qc.barrier(label="Encoding")
        self._encode_logical_state(qc, data, initial_state)
        
        # Step 2: QEC 라운드
        for r in range(self.num_rounds):
            qc.barrier(label=f"QEC Round {r+1}")
            self._measure_stabilizers(qc, data, ancilla_regs[r], syndrome_regs[r])
        
        # Step 3: 데이터 큐빗 최종 측정
        qc.barrier(label="Final Measurement")
        qc.measure(data, data_meas)
        
        return qc
    
    def _encode_logical_state(self, qc: QuantumCircuit, data: QuantumRegister, state: int):
        """
        논리적 |0⟩_L 또는 |1⟩_L을 인코딩합니다.
        
        |0⟩_L: C⊥ = [7,3,4] 코드워드의 균등 중첩
        |1⟩_L: X_L |0⟩_L
        
        Encoding (Reduced Systematic Generator):
          H(q0), H(q1), H(q2) → 3-bit 메시지 중첩 생성
          CNOT으로 패리티 비트 계산:
            q3 = q0⊕q1⊕q2
            q4 = q0⊕q1
            q5 = q0⊕q2
            q6 = q1⊕q2
        """
        if self.distance != 3:
            raise NotImplementedError("Encoding only implemented for d=3")
        
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
        return (
            f"=== Color Code Circuit Summary ===\n"
            f"Distance: {self.distance}\n"
            f"Data Qubits: {self.num_data}\n"
            f"Stabilizers: {self.num_stabilizers} ({self.num_stabilizers // 2} X-type + {self.num_stabilizers // 2} Z-type)\n"
            f"QEC Rounds: {self.num_rounds}\n"
            f"Ancilla per Round: {self.num_ancilla_per_round}\n"
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
    cc = ColorCodeCircuit(distance=3, num_rounds=1)
    print(cc.get_circuit_summary())
    
    qc = cc.build_circuit(initial_state=0)
    print(f"Circuit depth: {qc.depth()}")
    print(f"Gate counts: {dict(qc.count_ops())}")
    print(f"\n{qc.draw(output='text', fold=120)}")
    
    # 신드롬 인덱스 매핑 확인
    indices = cc.get_syndrome_indices()
    print(f"\nSyndrome indices: {indices}")
