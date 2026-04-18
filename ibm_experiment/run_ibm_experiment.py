"""
Phase 2: IBM Surface Code 실험 파이프라인

Supported code_types:
  - surface_code          : rotated surface code (Nighthawk / ibm_miami)
  - heavyhex_surface_code : depth-7 heavy-hex embedding (ibm_boston legacy)

사용법:
  python3 ibm_experiment/run_ibm_experiment.py
  python3 ibm_experiment/run_ibm_experiment.py --code heavyhex_surface_code
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

from simulators.ibm_simulator import IBMSimulator
from extractors.syndrome_extractor import SyndromeExtractor
from extractors.stim_compat import StimFormatConverter
from decoders.ml_decoder_adapter import MLDecoderAdapter
from decoders.hybrid_decoder import HybridMWPMDecoder
from evaluation.logical_error_rate import LogicalErrorRateEvaluator
from logger import log_to_file
from paths import ProjectPaths


# ============================================================================
# Circuit factory
# ============================================================================
def get_circuit_class(code_type: str):
    """Resolve circuit generator class for the requested code_type."""
    if code_type == "surface_code":
        from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
        return SurfaceCodeCircuit
    if code_type == "heavyhex_surface_code":
        from circuits.heavyhex_surface_code_depth7 import HeavyHexSurfaceCode
        return HeavyHexSurfaceCode
    raise ValueError(f"Unknown code_type: {code_type}")


def get_code_config(config: dict, code_type: str) -> dict:
    """Pick the per-code config block (surface_code / heavyhex_surface_code)."""
    if code_type in config:
        return config[code_type]
    raise KeyError(f"config is missing section '{code_type}'")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="IBM Phase 2 Experiment")
    parser.add_argument("-m", "--models", nargs="+", type=str, default=None,
                        help="실행할 모델 (미지정 시 config의 top_models)")
    parser.add_argument("-d", "--distance", nargs="+", type=int, default=None,
                        help="실행할 distance (미지정 시 config의 distances)")
    parser.add_argument("-n", "--noise", nargs="+", type=str, default=None,
                        help="노이즈 프로파일 목록 (미지정 시 stim config의 active_noise)")
    parser.add_argument("-b", "--backend", type=str, default=None,
                        help="실행할 IBM backend (미지정 시 config의 backend)")
    parser.add_argument("-i", "--instance", type=str, default=None,
                        help="Qiskit Runtime Service instance (미지정 시 config의 backend instance)")
    parser.add_argument("--code", type=str, default=None,
                        choices=["surface_code", "heavyhex_surface_code"],
                        help="code_type override (미지정 시 config의 evaluation.code_type)")
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


# ============================================================================
# Discord / CSV
# ============================================================================
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
                "description": f"**Weight**: `{wn_type}` (`{wn_params}`)\n"
                               f"**Setting**: `d={d}`, `p={p}`, Error: `{err_type}`",
                "color": 3447003,
                "fields": [
                    {"name": "🎯 LER", "value": f"{ler:.4f}", "inline": True},
                    {"name": "📊 Total Shots", "value": str(total_shots), "inline": True},
                    {"name": "🤖 Model", "value": model_name, "inline": True},
                ], "footer": {"text": f"STL Lab Server | IBM Phase 2 | {backend_name}"}}]
        }, timeout=5)
    except Exception:
        log_to_file(f"IBM | Failed to send Discord alert: {model_name} | d={d}, p={p}, "
                    f"{err_type} | {wn_params} | LER={ler:.4f}")


def save_results(results: list):
    output_dir = PATHS.experiment_result_dir("ibm")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"ibm_results_{timestamp}.csv")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    headers = ["Model", "Distance", "Num_Rounds", "Backend", "Shots",
               "Stim_Error_Rate", "Stim_Error_Type", "Weight_Noise",
               "Logical_Error_Rate", "Total_Shots", "Logical_Errors", "Timestamp"]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([r["model_name"], r["distance"], r["num_rounds"],
                r["backend"], r["shots"], r["stim_error_rate"], r["stim_error_type"],
                r["weight_noise"], f"{r['logical_error_rate']:.6f}", r["total_shots"],
                r["logical_errors"], timestamp])
    print(f"\n>>> Results saved to: {filepath}")


# ============================================================================
# Circuit construction
# ============================================================================
def build_circuit(code_type: str, CircuitClass, distance: int, num_rounds: int,
                  code_cfg: dict, backend_for_layout):
    """
    Construct (circuit_obj, qiskit_circuit, initial_layout, layout_diag)
    for the requested code_type. Performs automatic qubit selection when
    enabled.
    """
    initial_layout = None
    layout_diag = None

    if code_type == "surface_code":
        physical_qubits = None
        if code_cfg.get("auto_qubit_selection", False) and backend_for_layout is not None:
            from utils.nighthawk_layout import select_best_patch
            strategy = code_cfg.get("qubit_selection_strategy", "min_cx_error")
            layout = select_best_patch(
                backend=backend_for_layout,
                distance=distance,
                strategy=strategy,
            )
            physical_qubits = layout["physical_qubits"]
            initial_layout = layout["initial_layout"]
            layout_diag = layout["diagnostics"]
        sc = CircuitClass(distance=distance, num_rounds=num_rounds,
                          physical_qubits=physical_qubits)
    elif code_type == "heavyhex_surface_code":
        sc = CircuitClass(distance=distance, num_rounds=num_rounds)
    else:
        raise ValueError(f"Unsupported code_type: {code_type}")

    qc = sc.build_circuit(initial_state=code_cfg["logical_initial_state"])
    return sc, qc, initial_layout, layout_diag


def resolve_mwpm_decoder(code_type: str, distance: int, num_rounds: int,
                         noise_profile):
    """
    Return an MWPM decoder appropriate for the given code_type.

    For surface_code: stim-DEM-based `MWPMDecoder` (noise-dependent).
    For heavyhex_surface_code: the legacy lookup-based `MWPMHeavyHexDecoder`
    (noise-independent).
    """
    if code_type == "surface_code":
        from decoders.mwpm_decoder import MWPMDecoder
        return MWPMDecoder(distance=distance, rounds=num_rounds,
                           noise_profile=noise_profile)
    if code_type == "heavyhex_surface_code":
        from decoders.mwpm_heavyhex_decoder import MWPMHeavyHexDecoder
        return MWPMHeavyHexDecoder(distance=distance)
    raise ValueError(f"Unsupported code_type: {code_type}")


def run_mwpm_baseline(code_type: str, mwpm_decoder, syndromes, data_states,
                      converter):
    """
    Run MWPM and return (N, num_data) corrections. Bridges the two decoder
    interfaces (stim-based batch on detectors vs. legacy on raw HW syndromes).
    """
    if code_type == "surface_code":
        det = converter.hw_to_mwpm_detectors(syndromes, data_states)
        return mwpm_decoder.decode_batch(det)
    return mwpm_decoder.decode(syndromes, data_states)


# ============================================================================
# Pipeline
# ============================================================================
def run_pipeline(config: dict):
    backend_cfg = config["backend"]
    eval_cfg = config["evaluation"]

    code_type = ARGS.code if ARGS.code else eval_cfg["code_type"]
    code_cfg = get_code_config(config, code_type)

    CircuitClass = get_circuit_class(code_type)

    top_models = ARGS.models if ARGS.models else eval_cfg["top_models"]
    baselines = eval_cfg.get("baselines", ["No_Correction", "MWPM"])

    # noise 목록 결정
    if ARGS.noise:
        noise_list = ARGS.noise
    else:
        stim_config_path = os.path.join(root_dir, "stim_simulation", "config.json")
        with open(stim_config_path) as f:
            noise_list = json.load(f)["experiment"]["active_noise"]

    # DD 옵션
    dd_sequence = backend_cfg.get("dd_sequence", "XY4") \
        if backend_cfg.get("dynamical_decoupling", False) else None

    results = []
    distances = ARGS.distance if ARGS.distance else code_cfg["distances"]

    for distance in distances:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]
        BACKEND = ARGS.backend if ARGS.backend else backend_cfg["backend_name"]
        INSTANCE = ARGS.instance if ARGS.instance else backend_cfg["instance"]

        print(f"\n{'='*70}")
        print(f"  Phase 2: code={code_type}, d={distance}, rounds={num_rounds}")
        print(f"  Backend: {BACKEND}, instance={INSTANCE}")
        print(f"  Noise profiles: {noise_list}")
        print(f"  Models: {top_models}")
        print(f"  Baselines: {baselines}")
        print(f"  DD: {dd_sequence or 'off'}")
        print(f"{'='*70}")

        # === Runner 먼저 생성 (layout 결정을 위해 backend 필요) ===
        print(f"\n>>> [Step 1] Initializing IBM backend...")
        runner = IBMSimulator(
            backend_type=backend_cfg["type"],
            backend_instance=INSTANCE,
            backend_name=BACKEND
        )
        backend_for_layout = runner.backend if backend_cfg["type"] == "qpu" else None

        print(f"\n>>> [Step 2] Building {code_type} Circuit...")
        sc, qc, initial_layout, layout_diag = build_circuit(
            code_type, CircuitClass, distance, num_rounds, code_cfg,
            backend_for_layout,
        )
        print(sc.get_circuit_summary())
        print(f"    Qiskit Circuit: {qc.num_qubits} qubits, depth={qc.depth()}")
        if layout_diag is not None:
            print(f"    Qubit-selection diagnostics: {layout_diag}")

        print(f"\n>>> [Step 3] Running on IBM backend...")
        counts = runner.run(
            qc,
            shots=backend_cfg["shots"],
            initial_layout=initial_layout,
            dd_sequence=dd_sequence,
        )

        print(f"\n>>> [Step 4] Extracting syndromes and data states...")
        syn_indices = sc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)

        # === Noise-independent baselines: converter for first noise supplies
        #     stim_data_indices and enables MWPM/graph formatting ===
        first_edge_dir = PATHS.stim_data_dir(code_type, noise_list[0], "graph")
        first_converter = StimFormatConverter(
            distance=distance, num_rounds=num_rounds,
            edge_dir=first_edge_dir, code_type=code_type
        )
        stim_data_indices = first_converter.get_data_qubit_indices()

        evaluator = LogicalErrorRateEvaluator(
            logical_z=syn_indices["logical_z"],
            initial_logical_state=code_cfg["logical_initial_state"],
            stim_data_indices=stim_data_indices,
        )

        # No Correction
        if "No_Correction" in baselines:
            no_correction = np.zeros_like(data_states)
            nc_result = evaluator.evaluate(data_states, no_correction, shot_counts)
            nc_ler = nc_result["logical_error_rate"]
            print(f"\n📊 IBM No Correction: LER={nc_ler:.4f} "
                  f"({nc_result['logical_errors']}/{nc_result['total_shots']})")
            send_discord_alert("No_Correction", distance, 0, "N/A",
                               nc_ler, nc_result["total_shots"], BACKEND, "N/A")
            results.append({
                "model_name": "No_Correction", "distance": distance,
                "num_rounds": num_rounds, "backend": BACKEND,
                "shots": backend_cfg["shots"], "stim_error_rate": 0,
                "stim_error_type": "N/A", "weight_noise": "N/A",
                "logical_error_rate": nc_ler,
                "total_shots": nc_result["total_shots"],
                "logical_errors": nc_result["logical_errors"],
            })

        # MWPM baseline
        #   - surface_code: evaluated per noise profile (noise-dependent DEM).
        #   - heavyhex_surface_code: noise-independent; single evaluation.
        mwpm_available = False
        legacy_mwpm_decoder = None
        legacy_mwpm_corrections = None
        if "MWPM" in baselines and code_type == "heavyhex_surface_code":
            try:
                legacy_mwpm_decoder = resolve_mwpm_decoder(
                    code_type, distance, num_rounds, noise_list[0])
                legacy_mwpm_corrections = run_mwpm_baseline(
                    code_type, legacy_mwpm_decoder, syndromes, data_states,
                    first_converter)
                mwpm_eval = evaluator.evaluate(
                    data_states, legacy_mwpm_corrections, shot_counts)
                mwpm_ler = mwpm_eval["logical_error_rate"]
                print(f"\n📊 MWPM (HeavyHex): LER={mwpm_ler:.4f} "
                      f"({mwpm_eval['logical_errors']}/{mwpm_eval['total_shots']})")
                send_discord_alert("MWPM", distance, 0, "N/A",
                                   mwpm_ler, mwpm_eval["total_shots"],
                                   BACKEND, "N/A")
                results.append({
                    "model_name": "MWPM", "distance": distance,
                    "num_rounds": num_rounds, "backend": BACKEND,
                    "shots": backend_cfg["shots"], "stim_error_rate": 0,
                    "stim_error_type": "N/A", "weight_noise": "N/A",
                    "logical_error_rate": mwpm_ler,
                    "total_shots": mwpm_eval["total_shots"],
                    "logical_errors": mwpm_eval["logical_errors"],
                })
                mwpm_available = True
            except Exception as e:
                print(f"\n    ⚠️ MWPM (HeavyHex) decoder failed: {e}")
                log_to_file(f"IBM | MWPM | d={distance} | FAILED: {e}")

        # === Noise-dependent loop ===
        for noise in noise_list:
            print(f"\n>>> [Step 5] Noise profile: {noise}")

            edge_dir = PATHS.stim_data_dir(code_type, noise, "graph")
            converter = StimFormatConverter(
                distance=distance, num_rounds=num_rounds,
                edge_dir=edge_dir, code_type=code_type
            )

            # Per-noise MWPM for surface_code
            mwpm_per_noise = None
            if "MWPM" in baselines and code_type == "surface_code":
                try:
                    mwpm_per_noise = resolve_mwpm_decoder(
                        code_type, distance, num_rounds, noise)
                    mwpm_corr = run_mwpm_baseline(
                        code_type, mwpm_per_noise, syndromes, data_states,
                        converter)
                    mwpm_eval = evaluator.evaluate(
                        data_states, mwpm_corr, shot_counts)
                    mwpm_ler = mwpm_eval["logical_error_rate"]
                    print(f"\n📊 MWPM ({noise}): LER={mwpm_ler:.4f} "
                          f"({mwpm_eval['logical_errors']}/{mwpm_eval['total_shots']})")
                    send_discord_alert("MWPM", distance, 0, "N/A",
                                       mwpm_ler, mwpm_eval["total_shots"],
                                       BACKEND, noise)
                    results.append({
                        "model_name": "MWPM", "distance": distance,
                        "num_rounds": num_rounds, "backend": BACKEND,
                        "shots": backend_cfg["shots"], "stim_error_rate": 0,
                        "stim_error_type": "N/A", "weight_noise": noise,
                        "logical_error_rate": mwpm_ler,
                        "total_shots": mwpm_eval["total_shots"],
                        "logical_errors": mwpm_eval["logical_errors"],
                    })
                    mwpm_available = True
                except Exception as e:
                    print(f"\n    ⚠️ MWPM (surface_code) failed for {noise}: {e}")
                    log_to_file(f"IBM | MWPM | d={distance}, {noise} | FAILED: {e}")
                    mwpm_per_noise = None

            for model_name in top_models:
                model_type = eval_cfg["model_type_map"].get(model_name, "graph")

                for p in eval_cfg["stim_error_rates"]:
                    for err_type in eval_cfg["stim_error_types"]:
                        weight_path = PATHS.stim_weight(
                            code_type, noise, model_name, distance, p, err_type)

                        if not os.path.exists(weight_path):
                            print(f"\n    ⚠️ Weight not found: {weight_path}. Skipping.")
                            continue

                        print(f"\n    >>> {model_name} (trained: d={distance}, "
                              f"p={p}, {err_type}, noise={noise})")

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
                            log_to_file(f"IBM | {model_name} | d={distance}, p={p}, "
                                        f"{err_type}, {noise} | FAILED format: {e}")
                            continue

                        try:
                            decoder = MLDecoderAdapter(
                                model_name=model_name, weight_path=weight_path,
                                model_type=model_type, distance=distance,
                                input_shape=model_input_shape,
                            )
                        except Exception as e:
                            print(f"    ❌ Model load failed: {e}")
                            log_to_file(f"IBM | {model_name} | d={distance}, p={p}, "
                                        f"{err_type}, {noise} | FAILED model load: {e}")
                            continue

                        try:
                            corrections = decoder.decode(model_input, edge_index=edge_index)
                            print(f"        Corrections shape: {corrections.shape}")
                        except Exception as e:
                            print(f"    ❌ Inference failed: {e}")
                            log_to_file(f"IBM | {model_name} | d={distance}, p={p}, "
                                        f"{err_type}, {noise} | FAILED inference: {e}")
                            continue

                        eval_result = evaluator.evaluate(data_states, corrections, shot_counts)
                        ler = eval_result["logical_error_rate"]
                        print(f"        ✅ Logical Error Rate: {ler:.4f} "
                              f"({eval_result['logical_errors']}/{eval_result['total_shots']})")

                        send_discord_alert(model_name, distance, p, err_type,
                                           ler, eval_result["total_shots"], BACKEND, noise)

                        results.append({
                            "model_name": model_name, "distance": distance,
                            "num_rounds": num_rounds, "backend": BACKEND,
                            "shots": backend_cfg["shots"], "stim_error_rate": p,
                            "stim_error_type": err_type, "weight_noise": noise,
                            "logical_error_rate": ler,
                            "total_shots": eval_result["total_shots"],
                            "logical_errors": eval_result["logical_errors"],
                        })

                        # Hybrid MWPM+ML (only when legacy heavyhex decoder exists)
                        if mwpm_available and code_type == "heavyhex_surface_code":
                            try:
                                hybrid = HybridMWPMDecoder(
                                    distance=distance,
                                    ml_decoder=decoder,
                                    converter=converter,
                                    model_type=model_type,
                                )
                                hybrid_corrections = hybrid.decode(syndromes, data_states)
                                hybrid_eval = evaluator.evaluate(
                                    data_states, hybrid_corrections, shot_counts)
                                hybrid_ler = hybrid_eval["logical_error_rate"]
                                print(f"        ✅ Hybrid MWPM+{model_name}: "
                                      f"LER={hybrid_ler:.4f} "
                                      f"({hybrid_eval['logical_errors']}/{hybrid_eval['total_shots']})")
                                results.append({
                                    "model_name": f"MWPM+{model_name}",
                                    "distance": distance,
                                    "num_rounds": num_rounds,
                                    "backend": BACKEND,
                                    "shots": backend_cfg["shots"],
                                    "stim_error_rate": p,
                                    "stim_error_type": err_type,
                                    "weight_noise": noise,
                                    "logical_error_rate": hybrid_ler,
                                    "total_shots": hybrid_eval["total_shots"],
                                    "logical_errors": hybrid_eval["logical_errors"],
                                })
                                send_discord_alert(f"MWPM+{model_name}", distance, p,
                                                   err_type, hybrid_ler,
                                                   hybrid_eval["total_shots"],
                                                   BACKEND, noise)
                            except Exception as e:
                                print(f"        ⚠️ Hybrid MWPM+{model_name} failed: {e}")
                                log_to_file(f"IBM | MWPM+{model_name} | d={distance}, p={p}, "
                                            f"{err_type}, {noise} | FAILED: {e}")

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
