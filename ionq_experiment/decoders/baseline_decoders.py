"""
Baseline 디코더 + 비교 평가

Phase 2 결과의 의미를 해석하기 위한 Baseline을 제공합니다.

Baseline 1: No Correction (정정 없음)
  - 측정된 데이터 큐빗을 그대로 사용
  - "아무것도 안 하면 얼마나 나쁜가?"의 기준선

Baseline 2: Lookup Table Decoder (최적 디코딩)
  - d=3 Steane code의 모든 가능한 신드롬에 대해
    최적의 correction을 미리 계산해둔 테이블
  - 단일 에러까지 완벽 정정 가능
  - "이론적 최선"에 가까운 상한선

비교:
  No Correction LER > ML Decoder LER ≥ Lookup Table LER
  이 관계가 성립하면 ML 디코더가 의미 있게 작동하는 것
"""

import numpy as np
from typing import Tuple


# ==============================================================================
# Steane Code [[7,1,3]] Lookup Table
# ==============================================================================
# 
# 신드롬 → 에러 정정 매핑
#
# X-stabilizers (S1, S2, S3) → Z-에러 검출
#   S1 = X₀X₁X₂X₃  (Red)
#   S2 = X₁X₂X₄X₅  (Green)
#   S3 = X₂X₃X₅X₆  (Blue)
#
# Z-stabilizers (S4, S5, S6) → X-에러 검출
#   S4 = Z₀Z₁Z₂Z₃  (Red)
#   S5 = Z₁Z₂Z₄Z₅  (Green)
#   S6 = Z₂Z₃Z₅Z₆  (Blue)
#
# 신드롬 패턴 → 단일 큐빗 에러 매핑:
#   X-syndrome (S1,S2,S3) = (1,0,0) → Z-error on q0 or q3
#   X-syndrome (S1,S2,S3) = (1,1,0) → Z-error on q1
#   X-syndrome (S1,S2,S3) = (1,1,1) → Z-error on q2
#   X-syndrome (S1,S2,S3) = (0,1,0) → Z-error on q4
#   X-syndrome (S1,S2,S3) = (0,1,1) → Z-error on q5
#   X-syndrome (S1,S2,S3) = (0,0,1) → Z-error on q6
#   (Z-syndrome도 동일한 패턴으로 X-에러를 찾음)
#
# Parity Check Matrix H:
#   [[1,1,1,1,0,0,0],
#    [0,1,1,0,1,1,0],
#    [0,0,1,1,0,1,1]]
#
# 각 열이 해당 큐빗의 신드롬 패턴:
#   q0: (1,0,0), q1: (1,1,0), q2: (1,1,1), q3: (1,0,1)
#   q4: (0,1,0), q5: (0,1,1), q6: (0,0,1)
# ==============================================================================

# H 행렬 (3×7)
H_MATRIX = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1],
], dtype=np.int8)

# 단일 에러 신드롬 → 큐빗 인덱스 매핑
# key: (s1, s2, s3) 튜플, value: 에러 큐빗 인덱스
SINGLE_ERROR_TABLE = {}
for q in range(7):
    syndrome = tuple(H_MATRIX[:, q].tolist())
    SINGLE_ERROR_TABLE[syndrome] = q


class NoCorrection:
    """
    Baseline 1: 아무 정정도 하지 않음.
    모든 correction을 0으로 반환합니다.
    """
    
    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        Args:
            syndromes: (N, ...) - 사용하지 않음
            data_states: (N, num_data)
        Returns:
            corrections: (N, num_data) 전부 0
        """
        N, num_data = data_states.shape
        return np.zeros((N, num_data), dtype=np.int8)


class LookupTableDecoder:
    """
    Baseline 2: Steane Code [[7,1,3]] 전용 Lookup Table 디코더.
    
    단일 큐빗 에러를 완벽하게 정정합니다.
    2개 이상의 에러가 동시에 발생하면 잘못된 정정을 할 수 있지만,
    이는 d=3 코드의 이론적 한계입니다.
    
    동작 방식:
      1. 6개 신드롬 중 X-type 3개 → Z-에러 위치 추정
      2. 6개 신드롬 중 Z-type 3개 → X-에러 위치 추정  
      3. 두 에러를 합쳐서 correction 벡터 생성
    """
    
    def __init__(self):
        self.h_matrix = H_MATRIX
        self.single_error_table = SINGLE_ERROR_TABLE
    
    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        Args:
            syndromes: (N, num_rounds, 6) 또는 (N, 6)
                       [X-Red, X-Green, X-Blue, Z-Red, Z-Green, Z-Blue]
            data_states: (N, 7)
        Returns:
            corrections: (N, 7)
        """
        # 마지막 라운드의 신드롬만 사용 (여러 라운드면 마지막 것)
        if syndromes.ndim == 3:
            syn = syndromes[:, -1, :]  # (N, 6) 마지막 라운드
        else:
            syn = syndromes  # (N, 6)
        
        N = syn.shape[0]
        corrections = np.zeros((N, 7), dtype=np.int8)
        
        for i in range(N):
            # X-type 신드롬 (앞 3개) → Z-에러 검출
            x_syn = tuple(syn[i, :3].astype(int).tolist())
            
            # Z-type 신드롬 (뒤 3개) → X-에러 검출
            z_syn = tuple(syn[i, 3:].astype(int).tolist())
            
            # Z-에러 정정 (X-syndrome으로 찾음)
            if x_syn != (0, 0, 0):
                z_error_qubit = self.single_error_table.get(x_syn, None)
                if z_error_qubit is not None:
                    corrections[i, z_error_qubit] ^= 1
            
            # X-에러 정정 (Z-syndrome으로 찾음)
            if z_syn != (0, 0, 0):
                x_error_qubit = self.single_error_table.get(z_syn, None)
                if x_error_qubit is not None:
                    corrections[i, x_error_qubit] ^= 1
        
        return corrections


def run_baseline_comparison(syndromes: np.ndarray, data_states: np.ndarray,
                            shot_counts: np.ndarray, ml_corrections: dict,
                            logical_z: list, initial_state: int = 0) -> dict:
    """
    Baseline과 ML 디코더의 Logical Error Rate를 비교합니다.
    
    Args:
        syndromes: (N, num_rounds, 6)
        data_states: (N, 7)
        shot_counts: (N,)
        ml_corrections: {model_name: (N, num_output) corrections}
        logical_z: 논리적 Z 연산자 큐빗 인덱스
        initial_state: 초기 논리적 상태
    
    Returns:
        dict: {decoder_name: {"ler": float, "errors": int, "total": int}}
    """
    from evaluation.logical_error_rate import LogicalErrorRateEvaluator
    
    evaluator = LogicalErrorRateEvaluator(
        logical_z=logical_z,
        initial_logical_state=initial_state
    )
    
    results = {}
    total_shots = int(shot_counts.sum())
    
    # Baseline 1: No Correction
    no_corr = NoCorrection()
    no_corr_corrections = no_corr.decode(syndromes, data_states)
    no_corr_result = evaluator.evaluate(data_states, no_corr_corrections, shot_counts)
    results["No Correction"] = {
        "ler": no_corr_result["logical_error_rate"],
        "errors": no_corr_result["logical_errors"],
        "total": total_shots,
    }
    
    # Baseline 2: Lookup Table
    lookup = LookupTableDecoder()
    lookup_corrections = lookup.decode(syndromes, data_states)
    lookup_result = evaluator.evaluate(data_states, lookup_corrections, shot_counts)
    results["Lookup Table (Optimal d=3)"] = {
        "ler": lookup_result["logical_error_rate"],
        "errors": lookup_result["logical_errors"],
        "total": total_shots,
    }
    
    # ML Decoders
    for name, corrections in ml_corrections.items():
        # corrections가 data보다 크면 잘라냄
        if corrections.shape[1] > data_states.shape[1]:
            corrections = corrections[:, :data_states.shape[1]]
        
        ml_result = evaluator.evaluate(data_states, corrections, shot_counts)
        results[name] = {
            "ler": ml_result["logical_error_rate"],
            "errors": ml_result["logical_errors"],
            "total": total_shots,
        }
    
    return results


def print_comparison(results: dict):
    """비교 결과를 정렬하여 출력합니다."""
    print(f"\n{'='*60}")
    print(f"  Baseline Comparison (Logical Error Rate)")
    print(f"{'='*60}")
    print(f"  {'Decoder':<35s} {'LER':>8s} {'Errors':>8s} {'Total':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    
    # LER 기준 정렬 (낮은 순)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["ler"])
    
    for name, r in sorted_results:
        print(f"  {name:<35s} {r['ler']:>8.4f} {r['errors']:>8d} {r['total']:>8d}")
    
    # 해석 출력
    best_name = sorted_results[0][0]
    worst_name = sorted_results[-1][0]
    no_corr_ler = results.get("No Correction", {}).get("ler", 0)
    
    print(f"\n  Best:  {best_name} (LER={sorted_results[0][1]['ler']:.4f})")
    print(f"  Worst: {worst_name} (LER={sorted_results[-1][1]['ler']:.4f})")
    
    if no_corr_ler > 0:
        for name, r in sorted_results:
            if name != "No Correction":
                improvement = (1 - r["ler"] / no_corr_ler) * 100
                print(f"  {name}: {improvement:+.1f}% vs No Correction")
    
    print(f"{'='*60}")


# ==============================================================================
# 단독 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    print("=== Baseline Decoder Test ===")
    
    # 테스트 데이터
    syndromes = np.array([
        [[0, 0, 0, 0, 0, 0]],  # 에러 없음
        [[1, 1, 0, 0, 0, 0]],  # X-syn=(1,1,0) → q1에 Z-에러
        [[0, 0, 0, 1, 1, 0]],  # Z-syn=(1,1,0) → q1에 X-에러
    ], dtype=np.float32)
    
    data_states = np.array([
        [0, 0, 0, 0, 0, 0, 0],  # 정상
        [0, 1, 0, 0, 0, 0, 0],  # q1 flipped
        [0, 1, 0, 0, 0, 0, 0],  # q1 flipped
    ], dtype=np.int8)
    
    shot_counts = np.array([500, 300, 200])
    
    # No Correction
    nc = NoCorrection()
    nc_corr = nc.decode(syndromes, data_states)
    print(f"No Correction: {nc_corr}")
    
    # Lookup Table
    lt = LookupTableDecoder()
    lt_corr = lt.decode(syndromes, data_states)
    print(f"Lookup Table corrections: {lt_corr}")
    
    # 비교
    print(f"\nExpected: Lookup should correct q1 errors")
    print(f"  Shot 0 (no error): correction={lt_corr[0]} → corrected={data_states[0] ^ lt_corr[0]}")
    print(f"  Shot 1 (Z on q1):  correction={lt_corr[1]} → corrected={data_states[1] ^ lt_corr[1]}")
    print(f"  Shot 2 (X on q1):  correction={lt_corr[2]} → corrected={data_states[2] ^ lt_corr[2]}")
