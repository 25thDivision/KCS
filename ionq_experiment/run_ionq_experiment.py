"""
Phase 2: IonQ 노이즈 모델 시뮬레이션 파이프라인

시나리오 A: Stim에서 학습한 ML 디코더를 IonQ Aria-1 노이즈 환경에서 테스트

실행 흐름:
  1. Color Code 회로 생성 (Qiskit) 
  2. IonQ 시뮬레이터에서 실행 (noise_model="aria-1")
  3. 측정 결과에서 신드롬 + 데이터 상태 추출
  4. StimFormatConverter로 Phase 1 모델 입력 형태로 변환
  5. Phase 1 학습된 ML 모델로 에러 위치 추정
  6. Logical Error Rate 계산
"""

import os
import sys
import json
import csv
import numpy as np
from datetime import datetime

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
from evaluation.logical_error_rate import LogicalErrorRateEvaluator


def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def save_results(results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"phase2_results_{timestamp}.csv")
    
    headers = [
        "Model", "Distance", "Num_Rounds", "Noise_Model", "Shots",
        "Stim_Error_Rate", "Stim_Error_Type",
        "Logical_Error_Rate", "Total_Shots", "Logical_Errors",
        "Timestamp"
    ]
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([
                r["model_name"], r["distance"], r["num_rounds"],
                r["noise_model"], r["shots"],
                r["stim_error_rate"], r["stim_error_type"],
                f"{r['logical_error_rate']:.6f}",
                r["total_shots"], r["logical_errors"],
                timestamp
            ])
    
    print(f"\n>>> Results saved to: {filepath}")
    return filepath


def run_pipeline(config: dict):
    backend_cfg = config["backend"]
    code_cfg = config["color_code"]
    eval_cfg = config["evaluation"]
    
    results = []
    
    for distance in code_cfg["distances"]:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]
        
        print(f"\n{'='*70}")
        print(f"  Phase 2: d={distance}, rounds={num_rounds}, "
              f"noise={backend_cfg['noise_model']}")
        print(f"{'='*70}")
        
        # ==============================================================
        # Step 1: Qiskit Color Code 회로 생성
        # ==============================================================
        print(f"\n>>> [Step 1] Building Color Code Circuit...")
        
        cc = ColorCodeCircuit(distance=distance, num_rounds=num_rounds)
        print(cc.get_circuit_summary())
        
        qc = cc.build_circuit(initial_state=code_cfg["logical_initial_state"])
        print(f"    Qiskit Circuit: {qc.num_qubits} qubits, depth={qc.depth()}")
        
        # ==============================================================
        # Step 2: IonQ 시뮬레이터 실행
        # ==============================================================
        print(f"\n>>> [Step 2] Running on IonQ Simulator...")
        
        runner = IonQSimulator(
            backend_type=backend_cfg["type"],
            noise_model=backend_cfg["noise_model"]
        )
        
        counts = runner.run(qc, shots=backend_cfg["shots"])
        
        # ==============================================================
        # Step 3: 측정 결과에서 신드롬 + 데이터 추출
        # ==============================================================
        print(f"\n>>> [Step 3] Extracting syndromes and data states...")
        
        syn_indices = cc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)
        
        print(f"    Unique outcomes: {len(syndromes)}")
        print(f"    Total shots: {shot_counts.sum()}")
        print(f"    Syndromes shape: {syndromes.shape}")
        print(f"    Data states shape: {data_states.shape}")
        
        # ==============================================================
        # Step 4: Stim 호환 형태로 변환
        # ==============================================================
        print(f"\n>>> [Step 4] Converting to Stim-compatible format...")
        
        converter = StimFormatConverter(distance=distance, num_rounds=num_rounds)
        
        # Logical Error Rate 평가기
        evaluator = LogicalErrorRateEvaluator(
            logical_z=syn_indices["logical_z"],
            initial_logical_state=code_cfg["logical_initial_state"]
        )
        
        # ==============================================================
        # Step 5: ML 디코더 추론 + 평가
        # ==============================================================
        for model_name in eval_cfg["top_models"]:
            model_type = eval_cfg["model_type_map"].get(model_name, "graph")
            
            # ★ 핵심: converter에서 정확한 input_shape을 가져옴
            model_input_shape = converter.get_model_input_shape(model_type)
            
            weight_dir = os.path.join(current_dir, eval_cfg["stim_weight_dir"])
            
            for p in eval_cfg["stim_error_rates"]:
                for err_type in eval_cfg["stim_error_types"]:
                    weight_file = f"best_{model_name}_d{distance}_p{p}_{err_type}.pth"
                    weight_path = os.path.join(weight_dir, weight_file)
                    
                    if not os.path.exists(weight_path):
                        print(f"\n    ⚠️ Weight not found: {weight_file}. Skipping.")
                        continue
                    
                    print(f"\n    >>> {model_name} (trained: d={distance}, p={p}, {err_type})")
                    
                    # --- 5a. 신드롬을 모델 입력 형태로 변환 ---
                    try:
                        if model_type == "graph":
                            model_input, edge_index = converter.to_graph_format(syndromes)
                            print(f"        Input shape: {model_input.shape} (graph)")
                        else:
                            model_input = converter.to_image_format(syndromes)
                            edge_index = None
                            print(f"        Input shape: {model_input.shape} (image)")
                    except Exception as e:
                        print(f"    ❌ Format conversion failed: {e}")
                        continue
                    
                    # --- 5b. 모델 로드 (input_shape 명시 전달) ---
                    try:
                        decoder = MLDecoderAdapter(
                            model_name=model_name,
                            weight_path=weight_path,
                            model_type=model_type,
                            distance=distance,
                            input_shape=model_input_shape,
                        )
                    except Exception as e:
                        print(f"    ❌ Model load failed: {e}")
                        continue
                    
                    # --- 5c. 추론 ---
                    try:
                        corrections = decoder.decode(model_input, edge_index=edge_index)
                        print(f"        Corrections shape: {corrections.shape}")
                    except Exception as e:
                        print(f"    ❌ Inference failed: {e}")
                        continue
                    
                    # --- 6. Logical Error Rate 계산 ---
                    eval_result = evaluator.evaluate(
                        data_states, corrections, shot_counts
                    )
                    
                    ler = eval_result["logical_error_rate"]
                    print(f"        ✅ Logical Error Rate: {ler:.4f} "
                          f"({eval_result['logical_errors']}/{eval_result['total_shots']})")
                    
                    results.append({
                        "model_name": model_name,
                        "distance": distance,
                        "num_rounds": num_rounds,
                        "noise_model": backend_cfg["noise_model"],
                        "shots": backend_cfg["shots"],
                        "stim_error_rate": p,
                        "stim_error_type": err_type,
                        "logical_error_rate": ler,
                        "total_shots": eval_result["total_shots"],
                        "logical_errors": eval_result["logical_errors"],
                    })
    
    return results


def main():
    print("=" * 70)
    print("  Phase 2: IonQ Noise Model Simulation")
    print("  Scenario A: Transfer Learning Test")
    print("=" * 70)
    
    config = load_config()
    results = run_pipeline(config)
    
    if results:
        output_dir = os.path.join(current_dir, "results")
        save_results(results, output_dir)
        
        print(f"\n{'='*70}")
        print("  Results Summary")
        print(f"{'='*70}")
        for r in results:
            print(f"  {r['model_name']:20s} | d={r['distance']} r={r['num_rounds']} | "
                  f"p={r['stim_error_rate']} ({r['stim_error_type']}) | "
                  f"LER={r['logical_error_rate']:.4f}")
    else:
        print("\n⚠️ No results generated. Check model weights and configuration.")
    
    print(f"\n{'='*70}")
    print("  Phase 2 Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
