"""
_verify_nvidia_residual.py  (검증 전용)

NVIDIA LER이 3개 noise에서 0.031로 동일했던 원인 진단 (가설 a vs b).
- per-noise matcher/converter 정상 전달은 코드 점검(A)에서 (b) 배제 확정.
- 여기서는 (a) 정량 확인: 잔차 detector 통계 + shots=10000 LER 분리 여부.

기존 통합 코드 로직(IBMSimulator + StimFormatConverter + MWPMDecoder +
nvidia_to_correction_frame)을 그대로 재현. config 영구 변경 없이 shots만 override.
"""
import os
import sys
import json

import numpy as np
import torch

_IBM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_IBM_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _IBM_DIR)

from paths import ProjectPaths
from simulators.ibm_simulator import IBMSimulator
from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
from utils.nighthawk_layout import select_best_patch
from extractors.syndrome_extractor import SyndromeExtractor
from extractors.stim_compat import StimFormatConverter
from decoders.mwpm_decoder import MWPMDecoder
from evaluation.logical_error_rate import LogicalErrorRateEvaluator
from decoders.nvidia_predecoder import (
    load_nvidia_predecoder, kcs_to_nvidia_input, run_nvidia_predecoder,
    nvidia_to_correction_frame,
)

P = ProjectPaths()
D = int(os.environ.get("VERIFY_DISTANCE", "3"))
T = int(os.environ.get("VERIFY_ROUNDS", str(D)))  # 기본 rounds = distance
SHOTS = int(os.environ.get("VERIFY_SHOTS", "10000"))
bc = json.load(open(os.path.join(_IBM_DIR, "config.json")))["backend"]
noise_list = json.load(
    open(os.path.join(_ROOT, "stim_simulation", "config.json"))
)["experiment"]["active_noise"]

print(f"=== NVIDIA residual verify: d={D}, rounds={T}, shots={SHOTS} ===")
print(f"instance={bc['instance']} backend={bc['backend_name']}")
print(f"noise profiles: {noise_list}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1) simulator(노이즈=ibm_miami calib)로 syndrome 1회 생성 (shots override) ---
runner = IBMSimulator(backend_type="simulator",
                      backend_instance=bc["instance"],
                      backend_name=bc["backend_name"])
lay = select_best_patch(backend=runner.hw_backend, distance=D,
                        strategy="min_cx_error", verbose=False)
sc = SurfaceCodeCircuit(distance=D, num_rounds=T, physical_qubits=lay["physical_qubits"])
qc = sc.build_circuit(initial_state=0)
counts = runner.run(qc, shots=SHOTS, initial_layout=lay["initial_layout"],
                    distance=D, num_rounds=T)
ext = SyndromeExtractor(sc.get_syndrome_indices())
syndromes, data_states, shot_counts = ext.extract_from_counts(counts)
print(f"unique outcomes={len(shot_counts)}, total_shots={int(shot_counts.sum())}")

# --- 2) NVIDIA 모델 1회 로드 ---
model = load_nvidia_predecoder(P.nvidia_accurate_ckpt, distance=D, n_rounds=T,
                               device=device, model_id=4)

# --- 3) 3개 noise 각각: baseline/residual 통계 + MWPM/NVIDIA LER ---
summary = []
for noise in noise_list:
    print("\n" + "=" * 70)
    edge = P.stim_data_dir("surface_code", noise, "graph")
    conv = StimFormatConverter(distance=D, num_rounds=T, edge_dir=edge, code_type="surface_code")
    ev = LogicalErrorRateEvaluator(logical_z=sc.logical_z, initial_logical_state=0,
                                   stim_data_indices=conv.get_data_qubit_indices())
    mwpm = MWPMDecoder(distance=D, rounds=T, noise_profile=noise)
    base_det = conv.hw_to_mwpm_detectors(syndromes, data_states)

    # MWPM baseline LER (shot-count weighted)
    mwpm_corr = mwpm.decode_batch(base_det)
    mwpm_eval = ev.evaluate(data_states, mwpm_corr, shot_counts)

    # NVIDIA + 잔차 (verbose_stats로 통계 출력)
    nv_in = kcs_to_nvidia_input(syndromes, data_states, sc.x_stabilizers, sc.z_stabilizers,
                                D, T, basis="Z", device=device)
    nv_out = run_nvidia_predecoder(model, nv_in)
    nv_corr = nvidia_to_correction_frame(
        nvidia_out=nv_out, baseline_detectors=base_det, matcher=mwpm.matcher,
        x_stabilizers=sc.x_stabilizers, z_stabilizers=sc.z_stabilizers,
        distance=D, num_rounds=T, logical_z=sc.logical_z, num_data=sc.num_data,
        verbose_stats=True, stats_tag=f"noise={noise}",
    )
    nv_eval = ev.evaluate(data_states, nv_corr, shot_counts)

    print(f">>> {noise}")
    print(f"    MWPM_LER  ={mwpm_eval['logical_error_rate']:.5f} "
          f"({mwpm_eval['logical_errors']}/{mwpm_eval['total_shots']})")
    print(f"    NVIDIA_LER={nv_eval['logical_error_rate']:.5f} "
          f"({nv_eval['logical_errors']}/{nv_eval['total_shots']})")
    summary.append((noise, mwpm_eval["logical_error_rate"], nv_eval["logical_error_rate"],
                    nv_eval["logical_errors"], nv_eval["total_shots"]))

print("\n" + "#" * 70)
print("SUMMARY (shots=%d)" % SHOTS)
print("#" * 70)
for noise, m, n, ne, nt in summary:
    print(f"  {noise:42s} MWPM={m:.5f}  NVIDIA={n:.5f} ({ne}/{nt})")
nv_vals = [s[2] for s in summary]
print(f"\nNVIDIA LER distinct values: {sorted(set(nv_vals))}")
print(f"NVIDIA LER all identical?  {len(set(nv_vals)) == 1}")
# binomial resolution at this shot count
import math
p = nv_vals[0]
tot = summary[0][4]
res = math.sqrt(max(p, 1e-9) * (1 - p) / tot) if tot else float("nan")
print(f"binomial 1-sigma @ p={p:.4f}, N={tot}: ±{res:.5f}")
