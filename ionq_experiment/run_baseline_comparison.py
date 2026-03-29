"""
Phase 2 Baseline 비교 스크립트
"""

import os
import sys
import json
import numpy as np
import csv
import argparse
from datetime import datetime
from logger import log_to_file

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(root_dir)

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
from paths import ProjectPaths

PATHS = ProjectPaths(root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="IonQ Baseline Comparison")
    parser.add_argument("-n", "--noise", type=str, default=None,
                        help="노이즈 프로파일 (미지정 시 config의 noise)")
    parser.add_argument("-m", "--models", nargs="+", type=str, default=None,
                        help="실행할 모델 (미지정 시 config의 top_models)")
    return parser.parse_args()

ARGS = parse_args()

def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def save_baseline_csv(all_results, platform, noise):
    output_dir = PATHS.experiment_result_dir("ionq")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"baseline_results_{timestamp}.csv")
    headers = ["Method", "Distance", "Num_Rounds", "Platform",
               "Weight_Noise", "Logical_Error_Rate", "Total_Shots",
               "Logical_Errors", "Timestamp"]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in all_results:
            writer.writerow([r["method"], r["distance"], r["num_rounds"],
                platform, noise, f"{r['ler']:.6f}",
                r["total_shots"], r["logical_errors"], timestamp])
    print(f">>> Baseline results saved to: {filepath}")

def main():
    print("=" * 60)
    print("  Phase 2: Baseline Comparison")
    print("=" * 60)

    config = load_config()
    backend_cfg = config["backend"]
    code_cfg = config["color_code"]
    eval_cfg = config["evaluation"]

    code_type = eval_cfg["code_type"]
    noise = ARGS.noise if ARGS.noise else eval_cfg["noise"]
    top_models = ARGS.models if ARGS.models else eval_cfg["top_models"]

    all_results = []

    for distance in code_cfg["distances"]:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]

        print(f"\n>>> d={distance}, rounds={num_rounds}, noise={backend_cfg['noise_model']}")

        cc = ColorCodeCircuit(distance=distance, num_rounds=num_rounds)
        qc = cc.build_circuit(initial_state=code_cfg["logical_initial_state"])

        print(f"\n>>> Running IonQ simulator (shots={backend_cfg['shots']})...")
        runner = IonQSimulator(
            backend_type=backend_cfg["type"],
            noise_model=backend_cfg["noise_model"]
        )
        counts = runner.run(qc, shots=backend_cfg["shots"])

        syn_indices = cc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)
        print(f"    Shots: {shot_counts.sum()}, Unique outcomes: {len(syndromes)}")

        edge_dir = PATHS.stim_data_dir(code_type, noise, "graph")
        converter = StimFormatConverter(
            distance=distance, num_rounds=num_rounds,
            edge_dir=edge_dir
        )
        model_input_shape = converter.get_model_input_shape("graph")

        ml_corrections = {}

        for model_name in top_models:
            weight_path = PATHS.stim_weight(code_type, noise, model_name, distance, 0.01, "X")

            if not os.path.exists(weight_path):
                print(f"    ⚠️ {weight_path} not found. Skipping.")
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
                log_to_file(f"[BASELINE] {model_name} | d={distance} | FAILED: {e}")

        results = run_baseline_comparison(
            syndromes=syndromes,
            data_states=data_states,
            shot_counts=shot_counts,
            ml_corrections=ml_corrections,
            logical_z=syn_indices["logical_z"],
            initial_state=code_cfg["logical_initial_state"],
        )
        print_comparison(results)

        # 결과 수집
        for r in results:
            all_results.append({
                "method": r["method"],
                "distance": distance,
                "num_rounds": num_rounds,
                "ler": r["ler"],
                "total_shots": r["total_shots"],
                "logical_errors": r["logical_errors"],
            })
            log_to_file(f"[BASELINE] {r['method']} | d={distance} | weight={noise} | LER={r['ler']:.4f} | weight={noise}")

    if all_results:
        save_baseline_csv(all_results, backend_cfg["noise_model"], noise)

if __name__ == "__main__":
    main()