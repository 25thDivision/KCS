"""
IonQ 측정 결과(bitstring)에서 신드롬 벡터와 데이터 큐빗 상태를 추출합니다.

Qiskit의 bitstring convention: 역순 (q_n-1 ... q_1 q_0)
이를 올바른 순서로 변환하고, 기존 Stim 파이프라인과 호환되는 형태로 출력합니다.
"""

import numpy as np
from typing import Tuple


class SyndromeExtractor:
    """
    IonQ 측정 결과에서 신드롬과 데이터 큐빗 상태를 추출합니다.
    
    Parameters:
        syndrome_indices (dict): ColorCodeCircuit.get_syndrome_indices()의 반환값
    """
    
    def __init__(self, syndrome_indices: dict):
        self.info = syndrome_indices
        self.num_data = self.info["num_data"]
        self.num_stabilizers = self.info["num_stabilizers"]
        self.num_rounds = self.info["num_rounds"]
        self.syndrome_ranges = self.info["syndrome_ranges"]
        self.data_range = self.info["data_range"]
        self.logical_z = self.info["logical_z"]
    
    def extract_from_counts(self, counts: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Qiskit counts dict에서 신드롬, 데이터 상태, shot 수를 추출합니다.
        
        Args:
            counts: IonQ 실행 결과 {bitstring: count}
                    Qiskit bitstring format: "data_meas syn_rN ... syn_r1 syn_r0"
                    (공백으로 구분된 레지스터, 각 레지스터 내부는 역순)
        
        Returns:
            syndromes: (total_shots, num_rounds, num_stabilizers) 배열
            data_states: (total_shots, num_data) 배열
            shot_counts: (total_shots,) 각 unique outcome의 반복 횟수
        """
        all_syndromes = []
        all_data = []
        all_shot_counts = []
        
        for bitstring, count in counts.items():
            syndrome_rounds, data_bits = self._parse_bitstring(bitstring)
            
            all_syndromes.append(syndrome_rounds)
            all_data.append(data_bits)
            all_shot_counts.append(count)
        
        syndromes = np.array(all_syndromes)        # (unique_outcomes, num_rounds, num_stabilizers)
        data_states = np.array(all_data)            # (unique_outcomes, num_data)
        shot_counts = np.array(all_shot_counts)     # (unique_outcomes,)
        
        return syndromes, data_states, shot_counts
    
    def expand_to_shots(self, syndromes: np.ndarray, data_states: np.ndarray,
                        shot_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unique outcome을 shot 수만큼 복제하여 개별 shot 배열로 확장합니다.
        ML 모델 입력에 필요한 (N, ...) 형태를 만듭니다.
        
        Returns:
            expanded_syndromes: (total_shots, num_rounds, num_stabilizers)
            expanded_data: (total_shots, num_data)
        """
        expanded_syn = np.repeat(syndromes, shot_counts, axis=0)
        expanded_data = np.repeat(data_states, shot_counts, axis=0)
        return expanded_syn, expanded_data
    
    def _parse_bitstring(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Qiskit bitstring을 파싱하여 신드롬과 데이터 비트를 추출합니다.
        
        Qiskit convention:
          - 여러 레지스터가 공백으로 구분됨
          - 레지스터 순서: 뒤에서부터 (마지막 추가된 레지스터가 가장 왼쪽)
          - 각 레지스터 내부: 역순 (q_n-1 ... q_0)
          
        우리 회로 레지스터 순서 (추가 순서):
          syn_r0, syn_r1, ..., syn_rN, data_meas
        
        Qiskit bitstring에서는:
          "data_meas syn_rN ... syn_r1 syn_r0"
        """
        # 공백으로 분리된 경우
        if " " in bitstring:
            parts = bitstring.split(" ")
            # parts[0] = data_meas (마지막 추가된 레지스터가 왼쪽)
            # parts[1:] = syn_rN, ..., syn_r1, syn_r0 (역순)
            
            data_str = parts[0]
            syndrome_strs = list(reversed(parts[1:]))  # r0, r1, ..., rN 순서로 복원
            
        else:
            # 공백 없이 연결된 경우: 비트 수로 분리
            total_bits = len(bitstring)
            
            # 뒤에서부터 잘라냄 (Qiskit은 역순)
            data_str = bitstring[:self.num_data]
            remaining = bitstring[self.num_data:]
            
            syndrome_strs = []
            for r in range(self.num_rounds - 1, -1, -1):
                start = r * self.num_stabilizers
                end = start + self.num_stabilizers
                syn_str = remaining[start:end] if end <= len(remaining) else "0" * self.num_stabilizers
                syndrome_strs.append(syn_str)
            syndrome_strs = list(reversed(syndrome_strs))
        
        # Qiskit 역순 보정: 각 레지스터 내부의 비트를 뒤집음
        data_bits = np.array([int(b) for b in reversed(data_str)], dtype=np.int8)
        
        syndrome_rounds = []
        for syn_str in syndrome_strs:
            syn_bits = np.array([int(b) for b in reversed(syn_str)], dtype=np.int8)
            syndrome_rounds.append(syn_bits)
        
        syndrome_rounds = np.array(syndrome_rounds)  # (num_rounds, num_stabilizers)
        
        return syndrome_rounds, data_bits
    
    def compute_logical_value(self, data_state: np.ndarray) -> int:
        """
        데이터 큐빗 상태에서 논리적 Z 값을 계산합니다.
        
        Args:
            data_state: (num_data,) 배열
        
        Returns:
            0 또는 1 (논리적 큐빗의 Z 측정 결과)
        """
        parity = 0
        for q in self.logical_z:
            parity ^= int(data_state[q])
        return parity
    
    def to_flat_syndrome(self, syndromes: np.ndarray) -> np.ndarray:
        """
        (N, num_rounds, num_stabilizers) → (N, num_rounds * num_stabilizers)
        라운드를 평탄화하여 ML 모델 입력에 사용합니다.
        """
        N = syndromes.shape[0]
        return syndromes.reshape(N, -1)


# ==============================================================================
# 단독 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    # 테스트용 가짜 counts
    test_indices = {
        "syndrome_ranges": [(0, 6)],
        "data_range": (6, 13),
        "stabilizer_info": [],
        "logical_z": [0, 1, 4],
        "logical_x": [0, 2, 5],
        "num_data": 7,
        "num_stabilizers": 6,
        "num_rounds": 1,
    }
    
    extractor = SyndromeExtractor(test_indices)
    
    # Qiskit format: "data_meas syn_r0"
    test_counts = {
        "0000000 000000": 500,
        "0000001 000001": 300,
        "1100100 010010": 200,
    }
    
    syndromes, data_states, shot_counts = extractor.extract_from_counts(test_counts)
    
    print(f"Syndromes shape: {syndromes.shape}")
    print(f"Data states shape: {data_states.shape}")
    print(f"Shot counts: {shot_counts}")
    print(f"Total shots: {shot_counts.sum()}")
    
    # Logical value 테스트
    for i in range(len(data_states)):
        lv = extractor.compute_logical_value(data_states[i])
        print(f"  Outcome {i}: data={data_states[i]}, logical_Z={lv}, count={shot_counts[i]}")
