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
from decoders.hybrid_ml_mwpm_decoder import HybridMLMWPMDecoder
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
def log_result(model_name, d, p, err_type, ler, total_shots, backend_name, weight_noise):
    """개별 모델 결과를 파일 로그(logs/logger)에만 기록. Discord 전송은
    (distance, noise) 단위 묶음으로 send_discord_summary 가 담당."""
    try:
        _, wn_params = weight_noise.split('/')
    except (ValueError, AttributeError):
        wn_params = weight_noise
    log_to_file(f"IBM | {model_name} | d={d}, p={p}, {err_type} | {wn_params} | LER={ler:.4f}")


def send_discord_summary(distance, noise, backend_name, baseline_rows, noise_rows):
    """(distance, noise) 한 구간의 모든 모델 결과를 1개 Discord 알림(코드블록 표)으로 전송.

    baseline_rows: No_Correction 등 distance-레벨 baseline (매 noise 알림 상단 반복 표시).
    noise_rows:    이 noise 의 MWPM / NVIDIA / 8 models / hybrids.
    """
    if not DISCORD_WEBHOOK_URL:
        return
    rows = list(baseline_rows) + list(noise_rows)
    if not rows:
        return
    lines = []
    for r in rows:
        p = r.get("stim_error_rate", 0)
        et = r.get("stim_error_type", "N/A")
        tag = f"(p{p},{et})" if p else ""
        lines.append(
            f"{r['model_name']:<18}{tag:<12} LER={r['logical_error_rate']:.4f} "
            f"({r['logical_errors']}/{r['total_shots']})"
        )
    table = "\n".join(lines)
    desc = (f"**d={distance}** | noise=`{noise}` | `{backend_name}`\n"
            f"```\n{table}\n```")
    if len(desc) > 4000:  # Discord embed description 제한 (~4096)
        desc = desc[:3990] + "\n…```"
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={
            "content": f"🔬 **[IBM Phase 2] d={distance} | {noise}**",
            "embeds": [{
                "title": "📊 Surface Code Results (per distance/noise)",
                "description": desc,
                "color": 3447003,
                "footer": {"text": f"STL Lab Server | IBM Phase 2 | {backend_name}"},
            }],
        }, timeout=5)
    except Exception:
        log_to_file(f"IBM | Failed to send Discord summary: d={distance}, noise={noise}")


def save_results(results: list, backend_type: str = "qpu"):
    # simulator 모드 결과는 실제 QPU 결과와 섞이지 않도록 별도 하위 디렉토리에 저장.
    base_dir = PATHS.experiment_result_dir("ibm")
    if str(backend_type).lower() == "simulator":
        output_dir = os.path.join(base_dir, "simulator")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_dir
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
    dd_sequence = backend_cfg.get("dd_sequence", "XX4") \
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
            backend_name=BACKEND,
            capture=backend_cfg.get("capture", False),
        )
        backend_for_layout = runner.hw_backend  # simulator/qpu 모두 실제 IBM 백엔드를 layout 소스로

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
            optimization_level=backend_cfg.get("optimization_level", 2),
            distance=distance,
            num_rounds=num_rounds,
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

        # === NVIDIA Ising pre-decoder (zero-shot, surface_code 한정, distance당 1회 로드) ===
        # 잔차 디코딩은 per-noise MWPM matcher를 재사용하므로 noise 루프 안에서 평가.
        nvidia_model = None
        nvidia_device = None
        if code_type == "surface_code":
            try:
                import torch
                from decoders.nvidia_predecoder import load_nvidia_predecoder
                nvidia_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                nvidia_model = load_nvidia_predecoder(
                    weight_path=PATHS.nvidia_accurate_ckpt,
                    distance=distance, n_rounds=num_rounds,
                    device=nvidia_device, model_id=4,
                )
                print(f"\n>>> NVIDIA pre-decoder loaded (Accurate, model_id=4, device={nvidia_device})")
            except Exception as e:
                print(f"\n    ⚠️ NVIDIA pre-decoder load failed: {e}")
                log_to_file(f"IBM | NVIDIA | d={distance} | FAILED load: {e}")
                nvidia_model = None

        # distance-레벨 baseline (No_Correction, heavyhex MWPM) 수집 시작점.
        # 이 구간 결과는 매 noise Discord 요약 상단에 반복 표시된다.
        _baseline_start = len(results)

        # No Correction
        if "No_Correction" in baselines:
            no_correction = np.zeros_like(data_states)
            nc_result = evaluator.evaluate(data_states, no_correction, shot_counts)
            nc_ler = nc_result["logical_error_rate"]
            print(f"\n📊 IBM No Correction: LER={nc_ler:.4f} "
                  f"({nc_result['logical_errors']}/{nc_result['total_shots']})")
            log_result("No_Correction", distance, 0, "N/A",
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
                log_result("MWPM", distance, 0, "N/A",
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

        # distance-레벨 baseline 행 (No_Correction, heavyhex MWPM) — 매 noise 요약 상단 표시용
        distance_baseline_rows = list(results[_baseline_start:])

        # === Noise-dependent loop ===
        for noise in noise_list:
            print(f"\n>>> [Step 5] Noise profile: {noise}")

            _noise_start = len(results)  # 이 noise 구간 결과 수집 시작점

            edge_dir = PATHS.stim_data_dir(code_type, noise, "graph")
            converter = StimFormatConverter(
                distance=distance, num_rounds=num_rounds,
                edge_dir=edge_dir, code_type=code_type
            )

            # Per-noise MWPM for surface_code
            mwpm_per_noise = None
            mwpm_corr_for_hybrid = None
            if "MWPM" in baselines and code_type == "surface_code":
                try:
                    mwpm_per_noise = resolve_mwpm_decoder(
                        code_type, distance, num_rounds, noise)
                    mwpm_corr = run_mwpm_baseline(
                        code_type, mwpm_per_noise, syndromes, data_states,
                        converter)
                    mwpm_corr_for_hybrid = mwpm_corr
                    mwpm_eval = evaluator.evaluate(
                        data_states, mwpm_corr, shot_counts)
                    mwpm_ler = mwpm_eval["logical_error_rate"]
                    print(f"\n📊 MWPM ({noise}): LER={mwpm_ler:.4f} "
                          f"({mwpm_eval['logical_errors']}/{mwpm_eval['total_shots']})")
                    log_result("MWPM", distance, 0, "N/A",
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

            # === NVIDIA Ising pre-decoder (zero-shot) + 잔차 PyMatching (A1) ===
            #   - NVIDIA NN 은 noise-independent, 잔차 디코딩 matcher 는 per-noise MWPM 재사용.
            #   - correction frame (N, num_data) → 기존 evaluator 로 동일하게 LER 계산.
            if (code_type == "surface_code" and nvidia_model is not None
                    and mwpm_per_noise is not None):
                try:
                    from decoders.nvidia_predecoder import (
                        kcs_to_nvidia_input, run_nvidia_predecoder,
                        nvidia_to_correction_frame,
                    )
                    nv_in = kcs_to_nvidia_input(
                        syndromes=syndromes, data_states=data_states,
                        x_stabilizers=sc.x_stabilizers, z_stabilizers=sc.z_stabilizers,
                        distance=distance, n_rounds=num_rounds,
                        basis="Z", device=nvidia_device,
                    )
                    nv_out = run_nvidia_predecoder(nvidia_model, nv_in)
                    nv_baseline_det = converter.hw_to_mwpm_detectors(syndromes, data_states)
                    nv_corr = nvidia_to_correction_frame(
                        nvidia_out=nv_out, baseline_detectors=nv_baseline_det,
                        matcher=mwpm_per_noise.matcher,
                        x_stabilizers=sc.x_stabilizers, z_stabilizers=sc.z_stabilizers,
                        distance=distance, num_rounds=num_rounds,
                        logical_z=sc.logical_z, num_data=sc.num_data,
                        verbose_stats=True, stats_tag=f"d={distance} noise={noise}",
                    )
                    nv_eval = evaluator.evaluate(data_states, nv_corr, shot_counts)
                    nv_ler = nv_eval["logical_error_rate"]
                    print(f"\n📊 NVIDIA ({noise}): LER={nv_ler:.4f} "
                          f"({nv_eval['logical_errors']}/{nv_eval['total_shots']})")
                    log_result("NVIDIA", distance, 0, "N/A",
                                       nv_ler, nv_eval["total_shots"], BACKEND, noise)
                    results.append({
                        "model_name": "NVIDIA", "distance": distance,
                        "num_rounds": num_rounds, "backend": BACKEND,
                        "shots": backend_cfg["shots"], "stim_error_rate": 0,
                        "stim_error_type": "N/A", "weight_noise": noise,
                        "logical_error_rate": nv_ler,
                        "total_shots": nv_eval["total_shots"],
                        "logical_errors": nv_eval["logical_errors"],
                    })
                except Exception as e:
                    print(f"\n    ⚠️ NVIDIA pre-decoder failed for {noise}: {e}")
                    log_to_file(f"IBM | NVIDIA | d={distance}, {noise} | FAILED: {e}")

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

                        log_result(model_name, distance, p, err_type,
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

                        # Hybrid MWPM+ML
                        #   - heavyhex: 내장 MWPMHeavyHexDecoder 사용 (legacy 동작 유지)
                        #   - surface_code: 위에서 계산한 mwpm_corr를 주입
                        run_hybrid = mwpm_available and (
                            code_type == "heavyhex_surface_code"
                            or (code_type == "surface_code"
                                and mwpm_corr_for_hybrid is not None)
                        )
                        if run_hybrid:
                            try:
                                hybrid = HybridMWPMDecoder(
                                    distance=distance,
                                    ml_decoder=decoder,
                                    converter=converter,
                                    model_type=model_type,
                                    code_type=code_type,
                                )
                                hybrid_mwpm_input = (
                                    mwpm_corr_for_hybrid
                                    if code_type == "surface_code" else None
                                )
                                hybrid_corrections = hybrid.decode(
                                    syndromes, data_states,
                                    mwpm_corrections=hybrid_mwpm_input,
                                )
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
                                log_result(f"MWPM+{model_name}", distance, p,
                                                   err_type, hybrid_ler,
                                                   hybrid_eval["total_shots"],
                                                   BACKEND, noise)
                            except Exception as e:
                                print(f"        ⚠️ Hybrid MWPM+{model_name} failed: {e}")
                                log_to_file(f"IBM | MWPM+{model_name} | d={distance}, p={p}, "
                                            f"{err_type}, {noise} | FAILED: {e}")

                        # Hybrid ML+MWPM (역순)
                        #   - ML이 먼저, residual을 MWPM으로 정정
                        #   - heavyhex_surface_code, surface_code 모두 지원 (MWPM 가용 여부 무관)
                        try:
                            ml_mwpm = HybridMLMWPMDecoder(
                                distance=distance,
                                converter=converter,
                                code_type=code_type,
                            )
                            ml_mwpm_corrections = ml_mwpm.decode(
                                data_states=data_states,
                                ml_corrections=corrections,
                            )
                            ml_mwpm_eval = evaluator.evaluate(
                                data_states, ml_mwpm_corrections, shot_counts)
                            ml_mwpm_ler = ml_mwpm_eval["logical_error_rate"]
                            print(f"        ✅ Hybrid {model_name}+MWPM: "
                                  f"LER={ml_mwpm_ler:.4f} "
                                  f"({ml_mwpm_eval['logical_errors']}/{ml_mwpm_eval['total_shots']})")
                            results.append({
                                "model_name": f"{model_name}+MWPM",
                                "distance": distance,
                                "num_rounds": num_rounds,
                                "backend": BACKEND,
                                "shots": backend_cfg["shots"],
                                "stim_error_rate": p,
                                "stim_error_type": err_type,
                                "weight_noise": noise,
                                "logical_error_rate": ml_mwpm_ler,
                                "total_shots": ml_mwpm_eval["total_shots"],
                                "logical_errors": ml_mwpm_eval["logical_errors"],
                            })
                            log_result(f"{model_name}+MWPM", distance, p,
                                               err_type, ml_mwpm_ler,
                                               ml_mwpm_eval["total_shots"],
                                               BACKEND, noise)
                        except Exception as e:
                            print(f"        ⚠️ Hybrid {model_name}+MWPM failed: {e}")
                            log_to_file(f"IBM | {model_name}+MWPM | d={distance}, p={p}, "
                                        f"{err_type}, {noise} | FAILED: {e}")

            # (distance, noise) 묶음 Discord 요약 — baseline(No_Correction 등) + 이 noise 의 모든 모델/하이브리드
            noise_rows = results[_noise_start:]
            send_discord_summary(distance, noise, BACKEND,
                                 distance_baseline_rows, noise_rows)

    return results


def main():
    print("=" * 70)
    print("  Phase 2: IBM Surface Code Experiment")
    print("=" * 70)

    results = run_pipeline(CONFIG)

    if results:
        save_results(results, backend_type=CONFIG["backend"].get("type", "qpu"))
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
