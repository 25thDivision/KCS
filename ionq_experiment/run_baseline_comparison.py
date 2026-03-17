"""
Phase 2 Baseline 비교 스크립트

IonQ 시뮬레이터 결과를 재활용하여 Baseline과 ML 디코더를 비교합니다.
IonQ API를 다시 호출하지 않고, 이전 실행 결과를 사용합니다.

사용법:
  python3 ionq_experiment/run_baseline_comparison.py
"""

import os
import sys
import json
import numpy as np

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from circuits.qiskit_colorcode_generator import ColorCodeCircuit
from simulators.ionq_simulator import IonQSimulator
from extractors.syndrome_extractor import SyndromeExtractor
from extractors.stim_compat import StimFormatConverter
from decoders.ml_decoder_adapter import MLDecoderAdapter
from decoders.baseline_decoders import (
    NoCorrection, LookupTableDecoder, 
    run_baseline_comparison, print_comparison
)
from evaluation.logical_error_rate import LogicalErrorRateEvaluator


def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("  Phase 2: Baseline Comparison")
    print("  IonQ Aria-1 Noise Model + ML Decoders vs Baselines")
    print("=" * 60)
    
    config = load_config()
    backend_cfg = config["backend"]
    code_cfg = config["color_code"]
    eval_cfg = config["evaluation"]
    
    for distance in code_cfg["distances"]:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]
        
        print(f"\n>>> d={distance}, rounds={num_rounds}, noise={backend_cfg['noise_model']}")
        
        # 1. 회로 생성
        cc = ColorCodeCircuit(distance=distance, num_rounds=num_rounds)
        qc = cc.build_circuit(initial_state=code_cfg["logical_initial_state"])
        
        # 2. IonQ 실행
        print(f"\n>>> Running IonQ simulator (shots={backend_cfg['shots']})...")
        runner = IonQSimulator(
            backend_type=backend_cfg["type"],
            noise_model=backend_cfg["noise_model"]
        )
        counts = runner.run(qc, shots=backend_cfg["shots"])
        
        # 3. 신드롬 추출
        syn_indices = cc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)
        
        print(f"    Shots: {shot_counts.sum()}, Unique outcomes: {len(syndromes)}")
        
        # 4. Stim 호환 변환 + ML 추론
        converter = StimFormatConverter(distance=distance, num_rounds=num_rounds)
        model_input_shape = converter.get_model_input_shape("graph")
        
        ml_corrections = {}
        
        weight_dir = os.path.join(current_dir, eval_cfg["stim_weight_dir"])
        
        for model_name in eval_cfg["top_models"]:
            # 대표적으로 p=0.01로 학습한 모델 사용
            weight_file = f"best_{model_name}_d{distance}_p0.01_X.pth"
            weight_path = os.path.join(weight_dir, weight_file)
            
            if not os.path.exists(weight_path):
                print(f"    ⚠️ {weight_file} not found. Skipping.")
                continue
            
            try:
                graph_input, edge_index = converter.to_graph_format(syndromes)
                
                decoder = MLDecoderAdapter(
                    model_name=model_name,
                    weight_path=weight_path,
                    model_type="graph",
                    distance=distance,
                    input_shape=model_input_shape,
                )
                
                corrections = decoder.decode(graph_input, edge_index=edge_index)
                ml_corrections[f"ML: {model_name} (p=0.01)"] = corrections
                
            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
        
        # 5. Baseline 비교
        results = run_baseline_comparison(
            syndromes=syndromes,
            data_states=data_states,
            shot_counts=shot_counts,
            ml_corrections=ml_corrections,
            logical_z=syn_indices["logical_z"],
            initial_state=code_cfg["logical_initial_state"],
        )
        
        print_comparison(results)


if __name__ == "__main__":
    main()
