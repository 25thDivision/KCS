"""
Phase 2 Baseline 비교 스크립트 (IBM Surface Code)
"""

import os
import sys
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(root_dir)

from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
from simulators.ibm_simulator import IBMSimulator
from extractors.syndrome_extractor import SyndromeExtractor
from extractors.stim_compat import StimFormatConverter
from decoders.ml_decoder_adapter import MLDecoderAdapter
from decoders.baseline_decoders import (
    NoCorrection, run_baseline_comparison, print_comparison
)
from evaluation.logical_error_rate import LogicalErrorRateEvaluator
from paths import ProjectPaths

PATHS = ProjectPaths(root_dir)

def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("  Phase 2: IBM Baseline Comparison")
    print("=" * 60)

    config = load_config()
    backend_cfg = config["backend"]
    code_cfg = config["surface_code"]
    eval_cfg = config["evaluation"]

    code_type = eval_cfg["code_type"]
    noise = eval_cfg["noise"]

    for distance in code_cfg["distances"]:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]

        print(f"\n>>> d={distance}, rounds={num_rounds}, backend={backend_cfg['backend_name']}")

        sc = SurfaceCodeCircuit(distance=distance, num_rounds=num_rounds)
        qc = sc.build_circuit(initial_state=code_cfg["logical_initial_state"])

        print(f"\n>>> Running IBM backend (shots={backend_cfg['shots']})...")
        runner = IBMSimulator(
            backend_type=backend_cfg["type"],
            backend_name=backend_cfg["backend_name"]
        )
        counts = runner.run(qc, shots=backend_cfg["shots"])

        syn_indices = sc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)

        edge_dir = PATHS.stim_data_dir(code_type, noise, "graph")
        converter = StimFormatConverter(
            distance=distance, num_rounds=num_rounds,
            edge_dir=edge_dir
        )
        model_input_shape = converter.get_model_input_shape("graph")

        ml_corrections = {}

        for model_name in eval_cfg["top_models"]:
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
