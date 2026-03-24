"""
IBM Surface Code 측정 결과에서 신드롬과 데이터 큐빗 상태를 추출합니다.

Qiskit bitstring convention: 레지스터 역순, 비트 역순
  "data_meas syn_r2 syn_r1 syn_r0"
  각 레지스터 내부도 역순 (MSB first)
"""

import numpy as np
from typing import Tuple


class SyndromeExtractor:
    def __init__(self, syn_indices: dict):
        """
        Args:
            syn_indices: SurfaceCodeCircuit.get_syndrome_indices()의 반환값
        """
        self.num_data = syn_indices["num_data"]
        self.num_stabilizers = syn_indices["num_stabilizers"]
        self.num_rounds = syn_indices["num_rounds"]
        self.logical_z = syn_indices["logical_z"]

    def extract_from_counts(self, counts: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Qiskit 측정 결과에서 신드롬과 데이터 상태를 추출합니다.

        Args:
            counts: {bitstring: count} 형태의 Qiskit 측정 결과

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

        return syndromes, data_states, shot_counts

    def _parse_bitstring(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Qiskit bitstring을 파싱합니다.

        Qiskit 레지스터 순서 (공백으로 구분):
          "data_meas syn_rN-1 ... syn_r1 syn_r0"
          (마지막 레지스터가 왼쪽, 첫 레지스터가 오른쪽)

        실제로는: parts[0] = data_meas, parts[1] = syn_r(N-1), ..., parts[N] = syn_r0
        """
        parts = bitstring.split(" ")

        if len(parts) == self.num_rounds + 1:
            # 공백으로 구분된 경우
            data_str = parts[0]
            syn_strs = list(reversed(parts[1:]))  # syn_r0이 마지막 → 역순
        else:
            # 공백 없이 하나의 문자열
            full = bitstring.replace(" ", "")
            data_str = full[:self.num_data]
            remainder = full[self.num_data:]
            syn_strs = []
            for r in range(self.num_rounds):
                start = r * self.num_stabilizers
                end = start + self.num_stabilizers
                syn_strs.append(remainder[start:end])

        # Qiskit 비트 역순 처리
        data_bits = np.array([int(b) for b in reversed(data_str)], dtype=np.int8)

        syn_rounds = np.zeros((self.num_rounds, self.num_stabilizers), dtype=np.float32)
        for r, syn_str in enumerate(syn_strs):
            syn_rounds[r] = np.array([int(b) for b in reversed(syn_str)], dtype=np.float32)

        return syn_rounds, data_bits

    def compute_logical_value(self, data_state: np.ndarray) -> int:
        """데이터 큐빗 상태에서 논리적 Z 값을 계산합니다."""
        logical_val = 0
        for q in self.logical_z:
            logical_val ^= int(data_state[q])
        return logical_val
