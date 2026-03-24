"""
Image 데이터셋 생성

사용법:
  python3 generate_dataset_image.py
  python3 generate_dataset_image.py -c color_code -d 3 -e X
  python3 generate_dataset_image.py -c surface_code -n realistic/dp0_mf0.005_rf0.005_gd0.004 -d 3
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from common.mapper_image import SyndromeImageMapper
from paths import ProjectPaths

def parse_args():
    parser = argparse.ArgumentParser(description="Image Dataset Generator")
    parser.add_argument("-c", "--code", nargs="+", type=str, default=None)
    parser.add_argument("-n", "--noise", nargs="+", type=str, default=None)
    parser.add_argument("-d", "--distance", nargs="+", type=int, default=None)
    parser.add_argument("-e", "--error_type", nargs="+", type=str, default=None)
    parser.add_argument("--cpu", nargs="?", type=str, default=None, const=None, help="num_workers 조정 (generate 단계)")
    return parser.parse_args()

ARGS = parse_args()
PATHS = ProjectPaths(ROOT_DIR)

CONFIG = PATHS.load_stim_config()
EXP = CONFIG["experiment"]
NOISE_PROFILES = EXP["noise_profiles"]
NOISE_RATES = EXP["error_rates"]

CODE_TYPES = ARGS.code if ARGS.code else EXP["code_types"]
ACTIVE_NOISE = ARGS.noise if ARGS.noise else EXP["active_noise"]
DISTANCES = ARGS.distance if ARGS.distance else EXP["distances"]
ERROR_TYPES = ARGS.error_type if ARGS.error_type else EXP["error_types"]

TRAIN_SAMPLES = {3: 10000000, 5: 1000000, 7: 1000000}
TEST_SAMPLES = {3: 100000, 5: 100000, 7: 100000}
CHUNK_SIZE = 5000

if ARGS.cpu == "full":
    NUM_WORKERS = max(1, (os.cpu_count() - 8))
else:
    NUM_WORKERS = max(1, (os.cpu_count() // 2) - 1)

def get_generator(code_type):
    if code_type == "color_code":
        from generators.color_code import create_color_code_circuit, generate_dataset
        return create_color_code_circuit, generate_dataset
    elif code_type == "surface_code":
        from generators.surface_code import create_surface_code_circuit, generate_dataset
        return create_surface_code_circuit, generate_dataset
    else:
        raise ValueError(f"Unknown code_type: {code_type}")

KEYS = PATHS.load_keys()
DISCORD_WEBHOOK_URL = KEYS.get("discord_generation", "")

def send_discord_alert(code_type, noise, d, p, err_type, status, detail=""):
    if not DISCORD_WEBHOOK_URL:
        return
    import requests
    
    color_map = {"start": 3447003, "done": 5763719, "error": 15548997}
    emoji_map = {"start": "🚀", "done": "✅", "error": "❌"}
    
    # 1. '/'를 기준으로 분리하고 앞글자 대문자로 변환 (예외 처리 포함)
    try:
        noise_type, noise_params = noise.split('/')
        noise_type = noise_type.capitalize() 
    except ValueError:
        noise_type = noise.capitalize()
        noise_params = "N/A"
        
    status_emoji = emoji_map.get(status, '📊')
    status_color = color_map.get(status, 3447003)
    
    # 2. Embed 구성 (Title은 심플하게, 복잡한 설정은 Description으로)
    embed = {
        "title": f"{status_emoji} Dataset Generation: {status.upper()}",
        "description": f"**Code**: `{code_type}`\n**Noise**: `{noise_type}` (`{noise_params}`)",
        "color": status_color,
        "fields": [
            # 3. 주요 지표 3개를 한 줄에 나란히 배치
            {"name": "📏 Distance (d)", "value": str(d), "inline": True},
            {"name": "📉 Error Rate (p)", "value": str(p), "inline": True},
            {"name": "💥 Error Type", "value": err_type, "inline": True},
        ],
        "footer": {"text": f"STL Server | Status: {status.title()}"}
    }
    
    # Detail이 있을 경우 아래에 새 줄로 추가 (inline=False)
    if detail:
        embed["fields"].append({"name": "📝 Detail", "value": detail, "inline": False})
        
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=5)
    except:
        pass

def send_completion_alert(total_files, elapsed_min=None):
    if not DISCORD_WEBHOOK_URL:
        return
    import requests
    
    time_str = f"{elapsed_min:.1f} min" if elapsed_min else "Unknown"
    
    embed = {
        "title": "🎉 All Datasets Generated Successfully!",
        "color": 15258703, # 눈에 띄는 황금색/노란색
        "fields": [
            {"name": "📁 Total Files", "value": f"**{total_files}** files", "inline": True},
            {"name": "⏱️ Elapsed Time", "value": f"**{time_str}**", "inline": True}
        ],
        "footer": {"text": "STL Server | Generation Complete"}
    }
    
    try:
        # content로 멘션을 주거나 텍스트를 남기고, embeds에 카드를 추가
        requests.post(DISCORD_WEBHOOK_URL, json={
            "content": "✨ **데이터셋 생성이 모두 완료되었습니다!**",
            "embeds": [embed]
        }, timeout=5)
    except:
        pass

def _generate_chunk(code_type, d, p, err_type, count, np_):
    _, generate_dataset = get_generator(code_type)
    return generate_dataset(d, d, p, count, error_type=err_type,
                            meas_noise=np_["meas_flip"], reset_noise=np_["reset_flip"], gate_noise=np_["gate_depol"])

def generate_and_save(mapper, output_dir, code_type, d, p, err_type, total_samples, file_prefix, np_):
    num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
    print(f"       [{file_prefix.upper()}] ({err_type}, p={p}) -> {NUM_WORKERS} workers...")
    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(_generate_chunk)(code_type, d, p, err_type, min(CHUNK_SIZE, total_samples - i * CHUNK_SIZE), np_)
        for i in tqdm(range(num_chunks), desc="       Processing Chunks"))

    print("       -> Mapping to Images & Merging...")
    all_images, all_labels = [], []
    for raw, phys in results:
        all_images.append(mapper.map_to_images(raw))
        all_labels.append(phys)

    full_images = np.concatenate(all_images, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    file_name = f"{file_prefix}_d{d}_p{p}_{err_type}.npz"
    np.savez_compressed(os.path.join(output_dir, file_name), features=full_images, labels=full_labels)
    print(f"       Saved: {file_name} (Shape: {full_images.shape})")

def main():
    start_time = time.time()
    file_count = 0
    print(f"=== Generating Image Datasets ===")
    print(f"Code: {CODE_TYPES}, Noise: {ACTIVE_NOISE}, Dist: {DISTANCES}, Err: {ERROR_TYPES}")

    for code_type in CODE_TYPES:
        create_circuit, _ = get_generator(code_type)
        for noise in ACTIVE_NOISE:
            if noise not in NOISE_PROFILES:
                print(f"⚠️ Noise '{noise}' not in config. Skipping.")
                continue
            np_ = NOISE_PROFILES[noise]
            output_dir = PATHS.stim_data_dir(code_type, noise, "image")
            print(f"\n{'#'*60}\n# {code_type}/{noise}\n# Params: {np_}\n{'#'*60}")

            for d in DISTANCES:
                train_count, test_count = TRAIN_SAMPLES[d], TEST_SAMPLES[d]
                print(f"\n>>> d={d} (Train: {train_count}, Test: {test_count})")
                circuit = create_circuit(d, d, 0.001, meas_noise=np_["meas_flip"],
                                         reset_noise=np_["reset_flip"], gate_noise=np_["gate_depol"])
                mapper = SyndromeImageMapper(circuit)

                for p in NOISE_RATES:
                    for err_type in ERROR_TYPES:
                        send_discord_alert(code_type, noise, d, p, err_type, "start",
                                           f"Train: {train_count}, Test: {test_count}")
                        try:
                            generate_and_save(mapper, output_dir, code_type, d, p, err_type, train_count, "train", np_)
                            generate_and_save(mapper, output_dir, code_type, d, p, err_type, test_count, "test", np_)
                            file_count += 2
                            send_discord_alert(code_type, noise, d, p, err_type, "done")
                        except Exception as e:
                            print(f"    ❌ Error: {e}")
                            send_discord_alert(code_type, noise, d, p, err_type, "error", str(e))

    elapsed = (time.time() - start_time) / 60
    send_completion_alert(file_count, elapsed)
    print(f"\n=== Done! ({file_count} files, {elapsed:.1f} min) ===")

if __name__ == "__main__":
    main()