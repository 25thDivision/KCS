"""
Color Code MWPM Baseline 실험 스크립트

기존 run_ionq_experiment.py와 동일한 파이프라인으로:
  1. Color Code 회로 생성
  2. IonQ 시뮬레이터/하드웨어에서 실행
  3. 신드롬 추출
  4. MWPM (Restriction Decoder) + No Correction + Lookup Table 비교
  5. 결과 CSV 저장 + Discord 알림

사용법:
  python run_mwpm_colorcode.py                         # config.json 기본값
  python run_mwpm_colorcode.py --noise heavy            # noise profile 지정
  python run_mwpm_colorcode.py --shots 2000             # shot 수 지정
  python run_mwpm_colorcode.py --noise heavy --shots 5000
"""

import os
import sys
import csv
import json
import argparse
import requests
import numpy as np
from datetime import datetime

# === Path Setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from circuits.qiskit_colorcode_generator import ColorCodeCircuit
from simulators.ionq_simulator import IonQSimulator
from extractors.syndrome_extractor import SyndromeExtractor
from evaluation.logical_error_rate import LogicalErrorRateEvaluator
from decoders.baseline_decoders import NoCorrection, LookupTableDecoder
from decoders.mwpm_colorcode_decoder import MWPMColorCodeDecoder

try:
    from paths import PATHS
except ImportError:
    # Fallback: PATHS가 없으면 로컬 경로 사용
    class FallbackPaths:
        def experiment_result_dir(self, platform):
            return os.path.join(current_dir, "results")
    PATHS = FallbackPaths()

try:
    from logger import log_to_file
except ImportError:
    def log_to_file(msg):
        print(f"[LOG] {msg}")


# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Color Code MWPM Baseline Experiment")
parser.add_argument("--noise", type=str, default=None, help="Noise profile (e.g., standard, heavy, extreme)")
parser.add_argument("--shots", type=int, default=None, help="Number of shots (overrides config)")
args = parser.parse_args()


# === Config Loading ===
def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


# === Discord Alert ===
def send_discord_alert(decoder_name, distance, ler, total_shots, noise_model, noise_profile):
    """Discord 알림 전송"""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    
    try:
        requests.post(webhook_url, json={
            "content": f"**🎯 MWPM Baseline Experiment**",
            "embeds": [{
                "title": f"📊 Color Code MWPM Results",
                "description": (
                    f"**Decoder**: `{decoder_name}`\n"
                    f"**Setting**: `d={distance}`, noise=`{noise_profile}`"
                ),
                "color": 5763719,  # green
                "fields": [
                    {"name": "🎯 LER", "value": f"{ler:.4f}", "inline": True},
                    {"name": "📊 Total Shots", "value": str(total_shots), "inline": True},
                    {"name": "🤖 Decoder", "value": decoder_name, "inline": True},
                ],
                "footer": {"text": f"STL Lab Server | MWPM Baseline | {noise_model}"}
            }]
        }, timeout=5)
    except Exception:
        log_to_file(f"MWPM | Failed to send Discord alert: {decoder_name} | d={distance}")


# === Result Saving ===
def save_results(results: list, noise_profile: str):
    """결과를 CSV로 저장합니다."""
    output_dir = os.path.join(PATHS.experiment_result_dir("ionq"), noise_profile)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"mwpm_baseline_{timestamp}.csv")
    
    headers = [
        "Model", "Distance", "Num_Rounds", "Noise_Model", "Shots",
        "Stim_Error_Rate", "Stim_Error_Type", "Weight_Noise",
        "Logical_Error_Rate", "Total_Shots", "Logical_Errors", "Timestamp"
    ]
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([
                r["model_name"], r["distance"], r["num_rounds"],
                r["noise_model"], r["shots"], r["stim_error_rate"],
                r["stim_error_type"], r["weight_noise"],
                f"{r['logical_error_rate']:.6f}",
                r["total_shots"], r["logical_errors"], timestamp
            ])
    
    print(f"\n>>> Results saved to: {filepath}")
    return filepath


# === Main Pipeline ===
def run_mwpm_experiment():
    config = load_config()
    backend_cfg = config["backend"]
    code_cfg = config["color_code"]
    eval_cfg = config["evaluation"]
    
    # CLI overrides
    noise_profile = args.noise if args.noise else eval_cfg.get("noise", "standard")
    shots = args.shots if args.shots else backend_cfg["shots"]
    noise_model = backend_cfg["noise_model"]
    
    results = []
    
    for distance in code_cfg["distances"]:
        num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]
        initial_state = code_cfg["logical_initial_state"]
        
        print(f"\n{'='*70}")
        print(f"  MWPM Baseline Experiment: d={distance}, rounds={num_rounds}")
        print(f"  Noise: {noise_profile}, Backend: {noise_model}, Shots: {shots}")
        print(f"{'='*70}")
        
        # ----- Step 1: Build Circuit -----
        print(f"\n>>> [Step 1] Building Color Code Circuit...")
        cc = ColorCodeCircuit(distance=distance, num_rounds=num_rounds)
        print(cc.get_circuit_summary())
        qc = cc.build_circuit(initial_state=initial_state)
        print(f"    Qiskit Circuit: {qc.num_qubits} qubits, depth={qc.depth()}")
        
        # ----- Step 2: Run on IonQ -----
        print(f"\n>>> [Step 2] Running on IonQ ({noise_model})...")
        runner = IonQSimulator(
            backend_type=backend_cfg["type"],
            noise_model=noise_model
        )
        counts = runner.run(qc, shots=shots)
        print(f"    Got {len(counts)} unique outcomes from {shots} shots")
        
        # ----- Step 3: Extract Syndromes -----
        print(f"\n>>> [Step 3] Extracting syndromes and data states...")
        syn_indices = cc.get_syndrome_indices()
        extractor = SyndromeExtractor(syn_indices)
        syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)
        
        total_shots = int(shot_counts.sum())
        print(f"    Syndromes: {syndromes.shape}, Data: {data_states.shape}")
        print(f"    Total shots: {total_shots}")
        
        # ----- Step 4: Evaluate All Decoders -----
        print(f"\n>>> [Step 4] Evaluating decoders...")
        
        evaluator = LogicalErrorRateEvaluator(
            logical_z=syn_indices["logical_z"],
            initial_logical_state=initial_state
        )
        
        decoder_results = {}
        
        # (a) No Correction
        no_corr = NoCorrection()
        nc_corrections = no_corr.decode(syndromes, data_states)
        nc_eval = evaluator.evaluate(data_states, nc_corrections, shot_counts)
        decoder_results["No_Correction"] = nc_eval
        print(f"    No Correction:         LER={nc_eval['logical_error_rate']:.4f} "
              f"({nc_eval['logical_errors']}/{nc_eval['total_shots']})")
        
        # (b) Lookup Table
        lookup = LookupTableDecoder()
        lt_corrections = lookup.decode(syndromes, data_states)
        lt_eval = evaluator.evaluate(data_states, lt_corrections, shot_counts)
        decoder_results["Lookup_Table"] = lt_eval
        print(f"    Lookup Table:          LER={lt_eval['logical_error_rate']:.4f} "
              f"({lt_eval['logical_errors']}/{lt_eval['total_shots']})")
        
        # (c) MWPM (Restriction Decoder)
        mwpm = MWPMColorCodeDecoder()
        mwpm_corrections = mwpm.decode(syndromes, data_states)
        mwpm_eval = evaluator.evaluate(data_states, mwpm_corrections, shot_counts)
        decoder_results["MWPM_Restriction"] = mwpm_eval
        print(f"    MWPM (Restriction):    LER={mwpm_eval['logical_error_rate']:.4f} "
              f"({mwpm_eval['logical_errors']}/{mwpm_eval['total_shots']})")
        
        # (d) MWPM (fast batch) — 결과 일치 확인
        mwpm_fast_corrections = mwpm.decode_batch_fast(syndromes, data_states)
        mwpm_fast_eval = evaluator.evaluate(data_states, mwpm_fast_corrections, shot_counts)
        fast_match = np.array_equal(mwpm_corrections, mwpm_fast_corrections)
        print(f"    MWPM (fast batch):     LER={mwpm_fast_eval['logical_error_rate']:.4f} "
              f"(match={fast_match})")
        
        # ----- Step 5: Correction 일치 분석 -----
        print(f"\n>>> [Step 5] MWPM vs Lookup Table correction comparison...")
        
        mismatch_count = 0
        mismatch_shots = 0
        for i in range(len(syndromes)):
            if not np.array_equal(mwpm_corrections[i], lt_corrections[i]):
                mismatch_count += 1
                mismatch_shots += shot_counts[i]
        
        print(f"    Mismatched outcomes: {mismatch_count}/{len(syndromes)} "
              f"({mismatch_shots}/{total_shots} shots)")
        
        if mismatch_count > 0:
            print(f"    ⚠️ MWPM and Lookup Table differ on {mismatch_count} outcomes!")
            print(f"       This is expected for multi-error syndromes on noisy hardware.")
            
            # 상위 5개 mismatch 출력
            print(f"\n    Top mismatches (by shot count):")
            mismatches = []
            for i in range(len(syndromes)):
                if not np.array_equal(mwpm_corrections[i], lt_corrections[i]):
                    mismatches.append((i, shot_counts[i]))
            mismatches.sort(key=lambda x: -x[1])
            
            for idx, (i, cnt) in enumerate(mismatches[:5]):
                syn_str = syndromes[i, -1, :] if syndromes.ndim == 3 else syndromes[i]
                print(f"      [{idx+1}] syn={syn_str.astype(int)} count={cnt}")
                print(f"           MWPM correction:  {mwpm_corrections[i]}")
                print(f"           Lookup correction: {lt_corrections[i]}")
        else:
            print(f"    ✅ MWPM and Lookup Table give identical corrections for all outcomes.")
        
        # ----- Step 6: Summary -----
        print(f"\n>>> [Step 6] Summary")
        print(f"{'='*60}")
        print(f"  {'Decoder':<25s} {'LER':>8s} {'Errors':>8s} {'Total':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        
        sorted_decoders = sorted(decoder_results.items(), 
                                  key=lambda x: x[1]["logical_error_rate"])
        for name, r in sorted_decoders:
            print(f"  {name:<25s} {r['logical_error_rate']:>8.4f} "
                  f"{r['logical_errors']:>8d} {r['total_shots']:>8d}")
        
        print(f"{'='*60}")
        
        # ----- Collect results for CSV -----
        for dec_name, dec_eval in decoder_results.items():
            results.append({
                "model_name": dec_name,
                "distance": distance,
                "num_rounds": num_rounds,
                "noise_model": noise_model,
                "shots": shots,
                "stim_error_rate": 0,
                "stim_error_type": "N/A",
                "weight_noise": noise_profile,
                "logical_error_rate": dec_eval["logical_error_rate"],
                "total_shots": dec_eval["total_shots"],
                "logical_errors": dec_eval["logical_errors"],
            })
            
            # Discord alert
            send_discord_alert(
                dec_name, distance,
                dec_eval["logical_error_rate"],
                dec_eval["total_shots"],
                noise_model, noise_profile
            )
    
    # Save all results
    if results:
        save_results(results, noise_profile)
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  Color Code MWPM Baseline Experiment")
    print("  Restriction Decoder (Delfosse 2014) via PyMatching")
    print("=" * 70)
    
    run_mwpm_experiment()