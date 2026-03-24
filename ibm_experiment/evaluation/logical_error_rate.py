"""
Logical Error Rate 평가

모델 출력(Stim num_qubits 차원)에서 data qubit 위치만 추출하여 평가합니다.

stim_data_indices가 제공되면 해당 인덱스의 값만 추출합니다.
제공되지 않으면 기존처럼 앞에서 num_data개를 잘라냅니다.
"""

import numpy as np
from typing import List, Optional


class LogicalErrorRateEvaluator:
    def __init__(self, logical_z: list, initial_logical_state: int = 0,
                 stim_data_indices: Optional[List[int]] = None):
        """
        Args:
            logical_z: logical Z 연산자가 작용하는 data qubit 인덱스 (Qiskit 기준)
            initial_logical_state: 초기 논리 상태 (0 또는 1)
            stim_data_indices: 모델 출력에서 data qubit에 해당하는 Stim 인덱스 리스트
                               None이면 기존 truncation 방식 사용
        """
        self.logical_z = logical_z
        self.initial_logical_state = initial_logical_state
        self.stim_data_indices = stim_data_indices

    def evaluate(self, data_states: np.ndarray, corrections: np.ndarray,
                 shot_counts: np.ndarray) -> dict:
        N = len(data_states)
        num_data = data_states.shape[1]

        # 모델 출력에서 data qubit 위치만 추출
        if self.stim_data_indices is not None and corrections.shape[1] > num_data:
            print(f"    [LogicalErrorRate] Extracting data qubit indices {self.stim_data_indices} "
                  f"from corrections ({corrections.shape[1]} → {num_data})")
            corrections = corrections[:, self.stim_data_indices]
        elif corrections.shape[1] > num_data:
            print(f"    [LogicalErrorRate] Truncating corrections ({corrections.shape[1]} → {num_data})")
            corrections = corrections[:, :num_data]

        logical_errors = 0
        total_shots = 0

        for i in range(N):
            corrected = (data_states[i] ^ corrections[i]) % 2

            logical_val = 0
            for q in self.logical_z:
                logical_val ^= int(corrected[q])

            if logical_val != self.initial_logical_state:
                logical_errors += shot_counts[i]

            total_shots += shot_counts[i]

        ler = logical_errors / total_shots if total_shots > 0 else 0.0

        return {
            "logical_error_rate": ler,
            "logical_errors": int(logical_errors),
            "total_shots": int(total_shots),
        }