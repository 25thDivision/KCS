"""
Surface Code용 Baseline 디코더

1. NoCorrection: 아무 보정 없이 데이터 큐빗 그대로 평가
2. LookupTableDecoder: 신드롬 → 보정 매핑 (d=3에서만 실용적)
"""

import numpy as np
from collections import defaultdict


class NoCorrection:
    """아무 보정도 하지 않는 baseline"""
    def decode(self, syndromes, data_states) -> np.ndarray:
        return np.zeros_like(data_states, dtype=np.int8)


class LookupTableDecoder:
    """
    시뮬레이션 데이터로 Lookup Table을 구축하는 디코더.
    d=3 (9 data qubits) 에서만 실용적입니다.
    """
    def __init__(self):
        self.table = {}

    def build_table(self, syndromes, corrections):
        """신드롬 → 최빈 보정으로 테이블 구축"""
        correction_counts = defaultdict(lambda: defaultdict(int))

        for i in range(len(syndromes)):
            syn_key = tuple(syndromes[i].flatten().astype(int))
            corr_key = tuple(corrections[i].astype(int))
            correction_counts[syn_key][corr_key] += 1

        for syn_key, corr_dict in correction_counts.items():
            best_corr = max(corr_dict, key=corr_dict.get)
            self.table[syn_key] = np.array(best_corr, dtype=np.int8)

        print(f"    [LookupTable] Built with {len(self.table)} entries")

    def decode(self, syndromes, data_states) -> np.ndarray:
        corrections = np.zeros_like(data_states, dtype=np.int8)

        for i in range(len(syndromes)):
            syn_key = tuple(syndromes[i].flatten().astype(int))
            if syn_key in self.table:
                corrections[i] = self.table[syn_key]

        return corrections


def run_baseline_comparison(syndromes, data_states, shot_counts,
                            ml_corrections, logical_z, initial_state):
    """전체 baseline 비교를 실행합니다."""
    from evaluation.logical_error_rate import LogicalErrorRateEvaluator

    evaluator = LogicalErrorRateEvaluator(
        logical_z=logical_z,
        initial_logical_state=initial_state
    )

    results = {}

    # 1. No Correction
    no_corr = NoCorrection()
    corrections = no_corr.decode(syndromes, data_states)
    eval_result = evaluator.evaluate(data_states, corrections, shot_counts)
    results["No Correction"] = eval_result

    # 2. ML Decoders
    for name, corr in ml_corrections.items():
        eval_result = evaluator.evaluate(data_states, corr, shot_counts)
        results[name] = eval_result

    return results


def print_comparison(results):
    print(f"\n{'='*60}")
    print(f"  Baseline Comparison Results")
    print(f"{'='*60}")
    for name, r in results.items():
        ler = r["logical_error_rate"]
        total = r["total_shots"]
        errors = r["logical_errors"]
        print(f"  {name:30s} | LER={ler:.4f} ({errors}/{total})")
    print(f"{'='*60}")
