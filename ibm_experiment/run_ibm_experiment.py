"""
Phase 2: IBM Eagle Surface Code 실험 파이프라인

사용법:
  python3 ibm_experiment/run_ibm_experiment.py
  python3 ibm_experiment/run_ibm_experiment.py -m GraphMamba GraphTransformer
"""

import os
import sys
import json
import csv
import argparse
import requests
import numpy as np
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(root_dir)

from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
from simulators.ibm_simulator import IBMSimulator
from extractors.syndrome_extractor import SyndromeExtractor
from extractors.stim_compat import StimFormatConverter
from decoders.ml_decoder_adapter import MLDecoderAdapter
from evaluation.logical_error_rate import LogicalErrorRateEvaluator
from logger import log_to_file
from paths import ProjectPaths

def parse_args():
    parser = argparse.ArgumentParser(description="IBM Phase 2 Experiment")
    parser.add_argument("-m", "--models", nargs="+", type=str, default=None,
                        help="실행할 모델 (미지정 시 config의 top_models)")
    parser.add_argument("-n", "--noise", type=str, default=None,
                        help="노이즈 프로파일 (미지정 시 config의 noise)")
    return parser.parse_args()

ARGS = parse_args()
PATHS = ProjectPaths(root_dir)

def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()
KEYS = PATHS.load_keys()
DISCORD_WEBHOOK_URL = KEYS.get("discord_ibm", "")

def send_discord_alert(model_name, d, p, err_type, ler, total_shots, backend_name, weight_noise):
    try:
        wn_type, wn_params = weight_noise.split('/')
        wn_type = wn_type.capitalize()
    except ValueError:
        wn_type, wn_params = weight_noise, "N/A"
    
    log_to_file(f"IBM | {model_name} | d={d}, p={p}, {err_type} | {wn_params} | LER={ler:.4f}")
    
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={
        "content": f"🔬 **[IBM Phase 2] {model_name} Evaluated!**",
        "embeds": [{"title": f"📊 Surface Code Experiment Results",
            "description": f"**Weight**: `{wn_type}` (`{wn_params}`)\n**Setting**: `d={d}`, `p={p}`, Error: `{err_type}`",
            "color": 3447003,
            "fields": [
                {"name": "🎯 LER", "value": f"{ler:.4f}", "inline": True},
                {"name": "📊 Total Shots", "value": str(total_shots), "inline": True},
                {"name": "🤖 Model", "value": model_name, "inline": True},
            ], "footer": {"text": f"STL Lab Server | IBM Phase 2 | {backend_name}"}}]
        }, timeout=5)
    except:
        log_to_file(f"IBM | Failed to send Discord alert: {model_name} | d={d}, p={p}, {err_type} | {wn_params} | LER={ler:.4f}")
        pass


def save_results(results: list):
    output_dir = os.path.join(PATHS.experiment_result_dir("ibm"), results[0]["weight_noise"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"ibm_results_{timestamp}.csv")
    headers = ["Model", "Distance", "Num_Rounds", "Backend", "Shots",
               "Stim_Error_Rate", "Stim_Error_Type", "Weight_Noise",
               "Logical_Error_Rate", "Total_Shots", "Logical_Errors", "Timestamp"]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([r["model_name"], r["distance"], r["num_rounds"],
                r["backend"], r["shots"], r["stim_error_rate"], r["stim_error_type"],
                r["weight_noise"], f"{r['logical_error_rate']:.6f}", r["total_shots"], r["logical_errors"], timestamp])
    print(f"\n>>> Results saved to: {filepath}")

def run_pipeline(config: dict):
    backend_cfg = config["backend"]
    code_cfg = config["surface_code"]
    eval_cfg = config["evaluation"]

    code_type = eval_cfg["code_type"]

    top_models = ARGS.models if ARGS.models else eval_cfg["top_models"]
    noise = ARGS.noise if ARGS.noise else eval_cfg["noise"]
    
    results = []

    for distance in code_cfg["distances"]:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]

        print(f"\n{'='*70}")
        print(f"  Phase 2: d={distance}, rounds={num_rounds}, backend={backend_cfg['backend_name']}")
        print(f"  Models: {top_models}")
        print(f"{'='*70}")

        print(f"\n>>> [Step 1] Building Surface Code Circuit...")
        sc = SurfaceCodeCircuit(distance=distance, num_rounds=num_rounds)
        print(sc.get_circuit_summary())
        qc = sc.build_circuit(initial_state=code_cfg["logical_initial_state"])
        print(f"    Qiskit Circuit: {qc.num_qubits} qubits, depth={qc.depth()}")

        print(f"\n>>> [Step 2] Running on IBM backend...")
        runner = IBMSimulator(
            backend_type=backend_cfg["type"],
            backend_name=backend_cfg["backend_name"]
        )
        counts = runner.run(qc, shots=backend_cfg["shots"])

        print(f"\n>>> [Step 3] Extracting syndromes and data states...")
        syn_indices = sc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)

        print(f"\n>>> [Step 4] Converting to Stim-compatible format...")
        edge_dir = PATHS.stim_data_dir(code_type, noise, "graph")
        converter = StimFormatConverter(
            distance=distance, num_rounds=num_rounds,
            edge_dir=edge_dir
        )

        stim_data_indices = converter.get_data_qubit_indices()

        evaluator = LogicalErrorRateEvaluator(
            logical_z=syn_indices["logical_z"],
            initial_logical_state=code_cfg["logical_initial_state"],
            stim_data_indices=stim_data_indices,
        )
        
        # No Correction 측정 (동일 shot)
        no_correction = np.zeros_like(data_states)
        nc_result = evaluator.evaluate(data_states, no_correction, shot_counts)
        nc_ler = nc_result["logical_error_rate"]
        print(f"\n📊 IBM No Correction: LER={nc_ler:.4f} ({nc_result['logical_errors']}/{nc_result['total_shots']})")

        send_discord_alert("No_Correction", distance, 0, "N/A",
                        nc_ler, nc_result["total_shots"], backend_cfg["backend_name"], noise)
        
        results.append({
            "model_name": "No_Correction", "distance": distance,
            "num_rounds": num_rounds, "backend": backend_cfg["backend_name"],
            "shots": backend_cfg["shots"], "stim_error_rate": 0,
            "stim_error_type": "N/A", "weight_noise": noise,
            "logical_error_rate": nc_ler,
            "total_shots": nc_result["total_shots"],
            "logical_errors": nc_result["logical_errors"],
        })

        for model_name in top_models:
            model_type = eval_cfg["model_type_map"].get(model_name, "graph")

            for p in eval_cfg["stim_error_rates"]:
                for err_type in eval_cfg["stim_error_types"]:
                    weight_path = PATHS.stim_weight(code_type, noise, model_name, distance, p, err_type)

                    if not os.path.exists(weight_path):
                        print(f"\n    ⚠️ Weight not found: {weight_path}. Skipping.")
                        continue

                    print(f"\n    >>> {model_name} (trained: d={distance}, p={p}, {err_type})")

                    try:
                        if model_type == "graph":
                            model_input_shape = converter.get_model_input_shape("graph")
                            model_input, edge_index = converter.to_graph_format(syndromes)
                            print(f"        Input shape: {model_input.shape} (graph)")
                        else:
                            model_input_shape = converter.get_model_input_shape("image")
                            model_input = converter.to_image_format(syndromes)
                            edge_index = None
                            print(f"        Input shape: {model_input.shape} (image)")
                    except Exception as e:
                        print(f"    ❌ Format conversion failed: {e}")
                        log_to_file(f"IBM | {model_name} | d={distance}, p={p}, {err_type}, {noise} | FAILED format: {e}")
                        continue

                    try:
                        decoder = MLDecoderAdapter(
                            model_name=model_name, weight_path=weight_path,
                            model_type=model_type, distance=distance,
                            input_shape=model_input_shape,
                        )
                    except Exception as e:
                        print(f"    ❌ Model load failed: {e}")
                        log_to_file(f"IBM | {model_name} | d={distance}, p={p}, {err_type}, {noise} | FAILED model load: {e}")
                        continue

                    try:
                        corrections = decoder.decode(model_input, edge_index=edge_index)
                        print(f"        Corrections shape: {corrections.shape}")
                    except Exception as e:
                        print(f"    ❌ Inference failed: {e}")
                        log_to_file(f"IBM | {model_name} | d={distance}, p={p}, {err_type}, {noise} | FAILED inference: {e}")
                        continue

                    eval_result = evaluator.evaluate(data_states, corrections, shot_counts)
                    ler = eval_result["logical_error_rate"]
                    print(f"        ✅ Logical Error Rate: {ler:.4f} "
                          f"({eval_result['logical_errors']}/{eval_result['total_shots']})")

                    send_discord_alert(model_name, distance, p, err_type,
                                       ler, eval_result["total_shots"], backend_cfg["backend_name"], noise)

                    results.append({
                        "model_name": model_name, "distance": distance,
                        "num_rounds": num_rounds, "backend": backend_cfg["backend_name"],
                        "shots": backend_cfg["shots"], "stim_error_rate": p,
                        "stim_error_type": err_type, "weight_noise": noise, "logical_error_rate": ler,
                        "total_shots": eval_result["total_shots"],
                        "logical_errors": eval_result["logical_errors"],
                    })

    return results

def main():
    print("=" * 70)
    print("  Phase 2: IBM Surface Code Experiment")
    print("=" * 70)

    results = run_pipeline(CONFIG)

    if results:
        save_results(results)
        print(f"\n{'='*70}")
        print("  Results Summary")
        print(f"{'='*70}")
        for r in results:
            print(f"  {r['model_name']:20s} | d={r['distance']} r={r['num_rounds']} | "
                  f"p={r['stim_error_rate']} ({r['stim_error_type']}) | "
                  f"LER={r['logical_error_rate']:.4f}")

    print(f"\n{'='*70}")
    print("  Phase 2 Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()