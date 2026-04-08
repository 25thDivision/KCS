"""
IBM Surface Code 측정 결과에서 신드롬과 데이터 큐빗 상태를 추출합니다.

depth7 회로 지원:
  - depth7=False (기존): per-round 레지스터 → "data syn_rN-1 ... syn_r0"
  - depth7=True: 단일 syn 레지스터 → "data syn"
    - syn은 8*num_cycles+2 bits, 앞쪽 8*num_cycles만 사용
    - per-cycle 8 bits를 (num_cycles, 8)로 reshape

Qiskit bitstring convention: 레지스터 역순, 비트 역순
"""

import numpy as np
from typing import Tuple


class SyndromeExtractor:
    def __init__(self, syn_indices: dict):
        """
        Args:
            syn_indices: HeavyHexSurfaceCode.get_syndrome_indices()의 반환값
        """
        self.num_data = syn_indices["num_data"]
        self.num_stabilizers = syn_indices["num_stabilizers"]
        self.num_rounds = syn_indices["num_rounds"]
        self.logical_z = syn_indices["logical_z"]

        # depth7 전용
        self.depth7 = syn_indices.get("depth7", False)
        self.hw_syn_per_cycle = syn_indices.get("hw_syn_per_cycle", self.num_stabilizers)
        self.hw_syn_total = syn_indices.get("hw_syn_total",
                                             self.num_stabilizers * self.num_rounds)

    def extract_from_counts(self, counts: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Qiskit 측정 결과에서 신드롬과 데이터 상태를 추출합니다.

        Returns:
            syndromes: (N, num_rounds, num_stabilizers) 신드롬 배열
            data_states: (N, num_data) 데이터 큐빗 최종 측정값
            shot_counts: (N,) 각 outcome의 반복 횟수
        """
        N = len(counts)
        syndromes = np.zeros((N, self.num_rounds, self.num_stabilizers), dtype=np.float32)
        data_states = np.zeros((N, self.num_data), dtype=np.int8)
        shot_counts = np.zeros(N, dtype=np.int64)

        for i, (bitstring, count) in enumerate(counts.items()):
            syn_rounds, data_bits = self._parse_bitstring(bitstring)
            syndromes[i] = syn_rounds
            data_states[i] = data_bits
            shot_counts[i] = count

        total = shot_counts.sum()
        print(f"    Unique outcomes: {N}")
        print(f"    Total shots: {total}")
        print(f"    Syndromes shape: {syndromes.shape}")
        print(f"    Data states shape: {data_states.shape}")
        if self.depth7:
            print(f"    [depth7] HW syn bits per cycle: {self.hw_syn_per_cycle}")

        return syndromes, data_states, shot_counts

    def _parse_bitstring(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Qiskit bitstring을 파싱합니다.

        depth7=False: "data_meas syn_rN-1 ... syn_r1 syn_r0" (N+1 parts)
        depth7=True:  "data_meas syn" (2 parts, 단일 syn 레지스터)
        """
        parts = bitstring.split(" ")

        if self.depth7:
            return self._parse_depth7(parts)
        else:
            return self._parse_standard(parts)

    def _parse_standard(self, parts) -> Tuple[np.ndarray, np.ndarray]:
        """기존 per-round 레지스터 파싱."""
        if len(parts) == self.num_rounds + 1:
            data_str = parts[0]
            syn_strs = list(reversed(parts[1:]))
        else:
            full = "".join(parts)
            data_str = full[:self.num_data]
            remainder = full[self.num_data:]
            syn_strs = []
            for r in range(self.num_rounds):
                start = r * self.num_stabilizers
                end = start + self.num_stabilizers
                syn_strs.append(remainder[start:end])

        data_bits = np.array([int(b) for b in reversed(data_str)], dtype=np.int8)

        syn_rounds = np.zeros((self.num_rounds, self.num_stabilizers), dtype=np.float32)
        for r, syn_str in enumerate(syn_strs):
            syn_rounds[r] = np.array([int(b) for b in reversed(syn_str)], dtype=np.float32)

        return syn_rounds, data_bits

    def _parse_depth7(self, parts) -> Tuple[np.ndarray, np.ndarray]:
        """
        depth7 단일 syn 레지스터 파싱.

        Qiskit register 순서: 먼저 추가된 레지스터가 오른쪽.
        depth7 회로: QuantumCircuit(qr, cr_syn, cr_dat)
          → cr_syn 먼저 추가, cr_dat 나중
          → bitstring: "data_meas syn" (data가 왼쪽=나중 레지스터)

        syn 레지스터: hw_syn_total bits, 각 비트 Qiskit 역순
        per-cycle 8 bits 추출 후 (num_cycles, 8)로 reshape
        """
        if len(parts) == 2:
            data_str = parts[0]
            syn_str = parts[1]
        else:
            # 공백 없이 연결된 경우
            full = "".join(parts)
            data_str = full[:self.num_data]
            syn_str = full[self.num_data:]

        # Qiskit 비트 역순 처리
        data_bits = np.array([int(b) for b in reversed(data_str)], dtype=np.int8)
        syn_bits = np.array([int(b) for b in reversed(syn_str)], dtype=np.float32)

        # per-cycle 8 bits만 추출 (마지막 +2 deferred는 무시)
        num_cycles = self.num_rounds
        syn_rounds = np.zeros((num_cycles, self.num_stabilizers), dtype=np.float32)

        for c in range(num_cycles):
            start = c * self.hw_syn_per_cycle
            end = start + self.hw_syn_per_cycle
            if end <= len(syn_bits):
                syn_rounds[c] = syn_bits[start:end]

        return syn_rounds, data_bits

    def compute_logical_value(self, data_state: np.ndarray) -> int:
        """데이터 큐빗 상태에서 논리적 Z 값을 계산합니다."""
        logical_val = 0
        for q in self.logical_z:
            logical_val ^= int(data_state[q])
        return logical_val

    def to_flat_syndrome(self, syndromes: np.ndarray) -> np.ndarray:
        """(N, num_rounds, num_stabilizers) → (N, num_rounds * num_stabilizers)"""
        N = syndromes.shape[0]
        return syndromes.reshape(N, -1)
