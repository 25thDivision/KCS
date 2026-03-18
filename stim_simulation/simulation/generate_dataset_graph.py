import os
import sys
import json
import time
import numpy as np
import requests  # Discord 웹훅 통신용 추가
from tqdm import tqdm
from joblib import Parallel, delayed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)  # keys.json 경로 탐색용 추가
sys.path.append(CURRENT_DIR)

from generators.color_code import create_color_code_circuit, generate_dataset
from common.mapper_graph import SyndromeGraphMapper

# ==============================================================================
# Discord Alert 내장 기능
# ==============================================================================
def _load_webhook_url() -> str:
    key_file = os.path.join(ROOT_DIR, "keys.json")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                return json.load(f).get("discord_webhook_url", "")
        except:
            pass
    return ""

WEBHOOK_URL = _load_webhook_url()

def send_dataset_alert(data_type: str, code_type: str, measurement: str,
                       distance: int, p: float, err_type: str, status: str, detail: str = ""):
    if not WEBHOOK_URL: return
    
    emoji_map = {"start": "🚀", "done": "✅", "error": "❌"}
    emoji = emoji_map.get(status, "📊")
    
    dtype_str = data_type.capitalize()
    meas_str = measurement.capitalize()
    
    content = f"{emoji} [{dtype_str}] [{meas_str}] d={distance}, p={p}, {err_type}  → {status}"
    
    if detail:
        if status == "error":
            content += f"\n> **Error Detail**: {detail}"
        else:
            content += f" ({detail})"
            
    try:
        requests.post(WEBHOOK_URL, json={"content": content}, timeout=5)
    except:
        pass

def send_completion_alert(data_type: str, total_files: int, elapsed_min: float = None):
    if not WEBHOOK_URL: return
    try:
        dtype_str = data_type.capitalize()
        desc = f"Total files: {total_files}"
        if elapsed_min:
            desc += f" | Elapsed: {elapsed_min:.1f} min"
            
        message = {"content": f"🎉 **[{dtype_str}] All Datasets Generated!** {desc}"}
        requests.post(WEBHOOK_URL, json=message, timeout=5)
    except:
        pass

# ==============================================================================
# config.json 로드
# ==============================================================================
def load_config():
    config_path = os.path.join(PARENT_DIR, "config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()
EXP = CONFIG["experiment"]

DISTANCES = EXP["distances"]
NOISE_RATES = EXP["error_rates"]
ERROR_TYPES = EXP["error_types"]
CODE_TYPES = EXP["code_types"]
MEASUREMENTS = EXP["measurements"]
MEAS_NOISE_MAP = EXP["meas_noise"]

TRAIN_SAMPLES = {
    3: 10000000,
    5: 1000000,
    7: 1000000
}
TEST_SAMPLES = {
    3: 100000,
    5: 100000,
    7: 100000
}

CHUNK_SIZE = 5000
NUM_WORKERS = max(1, (os.cpu_count() // 2) - 1)

DATASET_BASE = os.path.join(PARENT_DIR, EXP["dataset_dir"])

# ==============================================================================
# 병렬 처리 함수
# ==============================================================================
def _generate_chunk(d, p, err_type, count, meas_noise):
    raw, phys = generate_dataset(d, d, p, count, error_type=err_type, meas_noise=meas_noise)
    return raw, phys

def generate_and_save(mapper, output_dir, d, p, err_type, total_samples, file_prefix, meas_noise):
    num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
    desc = f"       [{file_prefix.upper()}] ({err_type}, p={p})"
    print(f"{desc} -> Generating with {NUM_WORKERS} workers (meas_noise={meas_noise})...")

    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(_generate_chunk)(
            d, p, err_type,
            min(CHUNK_SIZE, total_samples - i * CHUNK_SIZE),
            meas_noise
        )
        for i in tqdm(range(num_chunks), desc="       Processing Chunks")
    )

    print("       -> Mapping to Graph Features & Merging...")

    all_features = []
    all_labels = []

    for raw_detectors, physical_errors in results:
        features = mapper.map_to_node_features(raw_detectors)
        all_features.append(features)
        all_labels.append(physical_errors)

    full_features = np.concatenate(all_features, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)

    file_name = f"{file_prefix}_d{d}_p{p}_{err_type}.npz"
    file_path = os.path.join(output_dir, file_name)

    np.savez_compressed(file_path, features=full_features, labels=full_labels)
    print(f"       Saved: {file_name} (Shape: {full_features.shape})")

# ==============================================================================
# 메인
# ==============================================================================
def main():
    start_time = time.time()
    file_count = 0

    print(f"=== Generating Graph Datasets (Parallel Mode: {NUM_WORKERS}) ===")
    print(f"Code Types: {CODE_TYPES}")
    print(f"Measurements: {MEASUREMENTS}")
    print(f"Distances: {DISTANCES}, Error Rates: {NOISE_RATES}, Error Types: {ERROR_TYPES}")

    for code_type in CODE_TYPES:
        for measurement in MEASUREMENTS:
            meas_noise = MEAS_NOISE_MAP[measurement]
            output_dir = os.path.join(DATASET_BASE, code_type, measurement, "graph")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n{'#'*60}")
            print(f"# {code_type}/{measurement} (meas_noise={meas_noise})")
            print(f"# Output: {output_dir}")
            print(f"{'#'*60}")

            for d in DISTANCES:
                train_count = TRAIN_SAMPLES[d]
                test_count = TEST_SAMPLES[d]
                print(f"\n>>> Distance d={d} (Train: {train_count}, Test: {test_count})")

                circuit = create_color_code_circuit(d, d, 0.001, meas_noise=meas_noise)
                mapper = SyndromeGraphMapper(circuit)

                edge_path = os.path.join(output_dir, f"edges_d{d}.npy")
                if not os.path.exists(edge_path):
                    np.save(edge_path, mapper.get_edges())
                    print(f"    - Saved Edges: {edge_path}")
                else:
                    print(f"    - Edges already exist: {edge_path}")

                for p in NOISE_RATES:
                    for err_type in ERROR_TYPES:
                        print(f"    -> Processing {err_type}-Error (p={p})...")
                        send_dataset_alert("graph", code_type, measurement, d, p, err_type, "start",
                                           f"Train: {train_count}, Test: {test_count}")
                        
                        try:
                            generate_and_save(mapper, output_dir, d, p, err_type, train_count, "train", meas_noise)
                            generate_and_save(mapper, output_dir, d, p, err_type, test_count, "test", meas_noise)
                            file_count += 2
                            send_dataset_alert("graph", code_type, measurement, d, p, err_type, "done",
                                               f"Train: {train_count}, Test: {test_count}")
                        except Exception as e:
                            print(f"    ❌ Error: {e}")
                            send_dataset_alert("graph", code_type, measurement, d, p, err_type, "error", str(e))

    elapsed = (time.time() - start_time) / 60
    send_completion_alert("graph", file_count, elapsed)
    print(f"\n=== All Graph Datasets Generated! ({file_count} files, {elapsed:.1f} min) ===")

if __name__ == "__main__":
    main()