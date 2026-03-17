"""
Logical Error Rate 평가

ML 디코더의 추정 결과를 적용한 후, 논리적 큐빗 상태가 보존되었는지 판정합니다.

평가 방식:
  1. 데이터 큐빗 최종 측정값에 ML 디코더의 correction을 XOR로 적용
  2. 정정된 상태에서 논리적 Z 연산자의 고유값을 계산
  3. 초기 논리적 상태와 비교하여 logical error 여부 판정

참고:
  Phase 1 모델은 Stim의 전체 큐빗(data + ancilla)에 대해 에러를 예측합니다.
  예: d=3 Stim → 10큐빗 출력, 하지만 실제 데이터 큐빗은 7개.
  따라서 correction 벡터에서 데이터 큐빗에 해당하는 앞부분만 사용합니다.
"""

import numpy as np
from typing import Tuple


class LogicalErrorRateEvaluator:
    """
    Logical Error Rate를 계산합니다.
    
    Parameters:
        logical_z (list): 논리적 Z 연산자의 support 큐빗 인덱스
        initial_logical_state (int): 초기 논리적 상태 (0 또는 1)
    """
    
    def __init__(self, logical_z: list, initial_logical_state: int = 0):
        self.logical_z = logical_z
        self.initial_state = initial_logical_state
    
    def evaluate(self, data_states: np.ndarray, corrections: np.ndarray,
                 shot_counts: np.ndarray = None) -> dict:
        """
        Logical Error Rate를 계산합니다.
        
        Args:
            data_states: (N, num_data) 데이터 큐빗 최종 측정값
            corrections: (N, num_model_output) ML 디코더가 추정한 에러 벡터
                         num_model_output >= num_data일 수 있음 (Stim 큐빗 수)
            shot_counts: (N,) 각 outcome의 반복 횟수 (None이면 모두 1)
        
        Returns:
            dict: {
                "logical_error_rate": float,
                "total_shots": int,
                "logical_errors": int,
                "logical_successes": int,
                "details": list
            }
        """
        N = len(data_states)
        num_data = data_states.shape[1]
        
        if shot_counts is None:
            shot_counts = np.ones(N, dtype=np.int64)
        
        # ★ 핵심: corrections가 data_states보다 크면 data 큐빗 부분만 사용
        if corrections.shape[1] > num_data:
            print(f"    [LogicalErrorRate] Correction dim ({corrections.shape[1]}) > "
                  f"data qubits ({num_data}). Truncating to first {num_data} positions.")
            corrections = corrections[:, :num_data]
        elif corrections.shape[1] < num_data:
            print(f"    [LogicalErrorRate] Warning: Correction dim ({corrections.shape[1]}) < "
                  f"data qubits ({num_data}). Padding with zeros.")
            padded = np.zeros((N, num_data), dtype=corrections.dtype)
            padded[:, :corrections.shape[1]] = corrections
            corrections = padded
        
        total_shots = int(shot_counts.sum())
        logical_errors = 0
        details = []
        
        for i in range(N):
            # 1. Correction 적용: 측정값 XOR 추정 에러
            corrected = (data_states[i] ^ corrections[i]).astype(np.int8)
            
            # 2. 논리적 Z 값 계산
            logical_z_value = 0
            for q in self.logical_z:
                logical_z_value ^= int(corrected[q])
            
            # 3. 초기 상태와 비교
            is_error = (logical_z_value != self.initial_state)
            
            if is_error:
                logical_errors += int(shot_counts[i])
            
            details.append({
                "data_state": data_states[i].tolist(),
                "correction": corrections[i].tolist(),
                "corrected_state": corrected.tolist(),
                "logical_z_value": logical_z_value,
                "is_logical_error": is_error,
                "shot_count": int(shot_counts[i]),
            })
        
        logical_successes = total_shots - logical_errors
        logical_error_rate = logical_errors / total_shots if total_shots > 0 else 0.0
        
        return {
            "logical_error_rate": logical_error_rate,
            "total_shots": total_shots,
            "logical_errors": logical_errors,
            "logical_successes": logical_successes,
            "details": details,
        }
    
    def compare_decoders(self, data_states: np.ndarray,
                         decoder_results: dict,
                         shot_counts: np.ndarray = None) -> dict:
        """
        여러 디코더의 성능을 비교합니다.
        
        Args:
            data_states: (N, num_data) 데이터 큐빗 측정값
            decoder_results: {decoder_name: corrections_array}
            shot_counts: (N,) 각 outcome의 반복 횟수
        
        Returns:
            dict: {decoder_name: evaluation_result}
        """
        comparison = {}
        for name, corrections in decoder_results.items():
            result = self.evaluate(data_states, corrections, shot_counts)
            comparison[name] = {
                "logical_error_rate": result["logical_error_rate"],
                "total_shots": result["total_shots"],
                "logical_errors": result["logical_errors"],
            }
        
        return comparison


# ==============================================================================
# 단독 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    print("=== Logical Error Rate Evaluator Test ===")
    
    evaluator = LogicalErrorRateEvaluator(logical_z=[0, 1, 4], initial_logical_state=0)
    
    # 테스트: corrections이 data보다 큰 경우 (Stim 10큐빗 → 7 data 큐빗)
    data_states = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int8)
    
    corrections = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10차원 출력
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int8)
    
    shot_counts = np.array([500, 500])
    
    result = evaluator.evaluate(data_states, corrections, shot_counts)
    print(f"LER: {result['logical_error_rate']:.4f}")
    print(f"Errors: {result['logical_errors']}/{result['total_shots']}")
