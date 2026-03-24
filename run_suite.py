"""
KCS 프로젝트 통합 워크플로우 관리자

사용법:
  # 전체 파이프라인 (데이터 생성 → 학습)
  nohup python3 run_suite.py --all -c color_code -d 3 --gpu0 CNN GCN --gpu1 GraphMamba GraphTransformer > suite.log 2>&1 &

  # 데이터 생성만
  python3 run_suite.py --generate -c color_code surface_code -d 3

  # 학습 (GPU 2장 병렬)
  python3 run_suite.py --train -c color_code -d 3 --gpu0 CNN GCN GAT APPNP --gpu1 GCNII GNN GraphTransformer GraphMamba

  # 학습 (GPU 1장)
  python3 run_suite.py --train -c surface_code -d 3 -g 0 -m GraphMamba GraphTransformer

  # 실험 (특정 모델만)
  python3 run_suite.py --experiment ionq -m GraphMamba GraphTransformer
  python3 run_suite.py --experiment ibm -m CNN GNN

  # 데이터 생성 + 학습 + 실험 한번에
  nohup python3 run_suite.py --all --experiment ionq -c color_code -d 3 -g 0 -m GraphMamba GraphTransformer > suite.log 2>&1 &

  # 상태 확인
  python3 run_suite.py --status
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STIM_DIR = os.path.join(ROOT_DIR, "stim_simulation")
STIM_SIM_DIR = os.path.join(STIM_DIR, "simulation")
IONQ_DIR = os.path.join(ROOT_DIR, "ionq_experiment")
IBM_DIR = os.path.join(ROOT_DIR, "ibm_experiment")
LOG_DIR = os.path.join(ROOT_DIR, "logs")


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd, cwd=None, log_file=None, background=False):
    cmd_str = " ".join(cmd)

    if background and log_file:
        ensure_log_dir()
        log_path = os.path.join(LOG_DIR, log_file)
        print(f"  🚀 [BG] {cmd_str}")
        print(f"     Log: {log_path}")
        
        # 핵심: 로그 파일을 열고, Popen으로 백그라운드 실행 후 객체 반환
        f = open(log_path, "w")
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)
        return process 
    else:
        print(f"  ▶️  {cmd_str}")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            print(f"  ❌ Failed with return code {result.returncode}")
        return result.returncode


# ==============================================================================
# 데이터 생성
# ==============================================================================
def do_generate(args):
    print(f"\n{'='*60}")
    print(f"  📊 Dataset Generation")
    print(f"{'='*60}")

    base_cmd_graph = ["python3", "generate_dataset_graph.py"]
    base_cmd_image = ["python3", "generate_dataset_image.py"]

    extra = []
    if args.code:
        extra += ["-c"] + args.code
    if args.noise:
        extra += ["-n"] + args.noise
    if args.distance:
        extra += ["-d"] + [str(d) for d in args.distance]
    if args.error_type:
        extra += ["-e"] + args.error_type
    if args.cpu:
        extra += ["--cpu"] + [args.cpu]

    print(f"\n  --- Graph datasets ---")
    run_cmd(base_cmd_graph + extra, cwd=STIM_SIM_DIR)

    print(f"\n  --- Image datasets ---")
    run_cmd(base_cmd_image + extra, cwd=STIM_SIM_DIR)

    print(f"\n  ✅ Dataset generation complete!")


# ==============================================================================
# ML 학습
# ==============================================================================
def do_train(args):
    print(f"\n{'='*60}")
    print(f"  🧠 ML Training")
    print(f"{'='*60}")

    base_cmd = ["python3", os.path.join(STIM_DIR, "run_stim_simulation.py")]

    # GPU 2장 병렬 모드
    if args.gpu0 or args.gpu1:
        processes = []  # 실행 중인 프로세스들을 담을 바구니

        if args.gpu0:
            cmd0 = base_cmd + ["-g", "0", "-m"] + args.gpu0
            if args.code:
                cmd0 += ["-c"] + args.code
            if args.distance:
                cmd0 += ["-d"] + [str(d) for d in args.distance]
            log0 = f"train_gpu0_{timestamp()}.log"
            
            # run_cmd의 결과를 p0로 받아서 리스트에 추가
            p0 = run_cmd(cmd0, cwd=ROOT_DIR, log_file=log0, background=True)
            if p0: processes.append(p0)

        if args.gpu1:
            cmd1 = base_cmd + ["-g", "1", "-m"] + args.gpu1
            if args.code:
                cmd1 += ["-c"] + args.code
            if args.distance:
                cmd1 += ["-d"] + [str(d) for d in args.distance]
            log1 = f"train_gpu1_{timestamp()}.log"
            
            # run_cmd의 결과를 p1로 받아서 리스트에 추가
            p1 = run_cmd(cmd1, cwd=ROOT_DIR, log_file=log1, background=True)
            if p1: processes.append(p1)

        print(f"\n  ⏳ Waiting for parallel training to finish...")
        
        # 핵심: 리스트에 담긴 모든 프로세스가 끝날 때까지 여기서 대기!
        for p in processes:
            p.wait()
            
        print(f"  ✅ Parallel training complete!")
        return

    # 단일 GPU 모드
    cmd = base_cmd[:]
    if args.gpu is not None:
        cmd += ["-g", str(args.gpu)]
    if args.models:
        cmd += ["-m"] + args.models
    if args.code:
        cmd += ["-c"] + args.code
    if args.noise:
        cmd += ["-n"] + args.noise
    if args.distance:
        cmd += ["-d"] + [str(d) for d in args.distance]

    if args.background:
        log = f"train_{timestamp()}.log"
        run_cmd(cmd, cwd=ROOT_DIR, log_file=log, background=True)
        print(f"\n  ✅ Training launched in background!")
    else:
        run_cmd(cmd, cwd=ROOT_DIR)
        print(f"\n  ✅ Training complete!")


# ==============================================================================
# 실험 (IonQ / IBM)
# ==============================================================================
def do_experiment(args):
    print(f"\n{'='*60}")
    print(f"  🔬 Hardware Experiment")
    print(f"{'='*60}")

    valid = ["ionq", "ibm", "ionq-baseline", "ibm-baseline"]
    cmd_map = {
        "ionq": os.path.join(IONQ_DIR, "run_ionq_experiment.py"),
        "ibm": os.path.join(IBM_DIR, "run_ibm_experiment.py"),
        "ionq-baseline": os.path.join(IONQ_DIR, "run_baseline_comparison.py"),
        "ibm-baseline": os.path.join(IBM_DIR, "run_baseline_comparison.py"),
    }

    for platform in args.experiment:
        if platform not in valid:
            print(f"  ❌ Unknown: {platform}. Options: {valid}")
            continue

        print(f"\n  --- {platform} ---")
        cmd = ["python3", cmd_map[platform]]

        if args.models:
            cmd += ["-m"] + args.models

        if args.background:
            log = f"experiment_{platform}_{timestamp()}.log"
            run_cmd(cmd, cwd=ROOT_DIR, log_file=log, background=True)
        else:
            run_cmd(cmd, cwd=ROOT_DIR)

    print(f"\n  ✅ All experiments complete!")


# ==============================================================================
# 상태 확인
# ==============================================================================
def do_status(args):
    print(f"\n{'='*60}")
    print(f"  📋 Project Status")
    print(f"{'='*60}")

    # GPU 상태
    print(f"\n  --- GPU Status ---")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"])

    # 실행 중인 프로세스
    print(f"\n  --- Running Processes ---")
    subprocess.run(["bash", "-c", "ps aux | grep 'python3.*stim\\|python3.*ionq\\|python3.*ibm' | grep -v grep"])

    # 데이터셋 현황
    print(f"\n  --- Datasets ---")
    dataset_dir = os.path.join(STIM_DIR, "dataset")
    if os.path.exists(dataset_dir):
        for code in sorted(os.listdir(dataset_dir)):
            code_path = os.path.join(dataset_dir, code)
            if os.path.isdir(code_path):
                for root, dirs, files in os.walk(code_path):
                    npz_files = [f for f in files if f.endswith('.npz')]
                    if npz_files:
                        rel = os.path.relpath(root, dataset_dir)
                        print(f"    {rel}: {len(npz_files)} files")

    # 가중치 현황
    print(f"\n  --- Saved Weights ---")
    weight_dir = os.path.join(STIM_DIR, "saved_weights")
    if os.path.exists(weight_dir):
        for root, dirs, files in os.walk(weight_dir):
            pth_files = [f for f in files if f.endswith('.pth')]
            if pth_files:
                rel = os.path.relpath(root, weight_dir)
                print(f"    {rel}: {len(pth_files)} files")

    # 최근 로그
    print(f"\n  --- Recent Logs ---")
    if os.path.exists(LOG_DIR):
        logs = sorted(os.listdir(LOG_DIR), reverse=True)[:5]
        for log in logs:
            log_path = os.path.join(LOG_DIR, log)
            size = os.path.getsize(log_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime("%m/%d %H:%M")
            print(f"    {log} ({size//1024}KB, {mtime})")
    else:
        print(f"    No logs yet.")


# ==============================================================================
# CLI 파싱
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="KCS Project Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_suite.py --generate -c color_code -d 3
  python3 run_suite.py --train -g 0 -m GraphMamba -c color_code -d 3
  python3 run_suite.py --train --gpu0 CNN GCN --gpu1 GraphMamba GraphTransformer -c surface_code -d 3
  python3 run_suite.py --experiment ionq -m GraphMamba GraphTransformer
  python3 run_suite.py --experiment ibm -m CNN GNN
  python3 run_suite.py --all --experiment ionq -c color_code -d 3 -m GraphMamba GraphTransformer
  python3 run_suite.py --status
        """
    )

    # 단계 선택
    stage = parser.add_argument_group("Stages")
    stage.add_argument("--generate", action="store_true", help="데이터 생성")
    stage.add_argument("--train", action="store_true", help="ML 학습")
    stage.add_argument("--experiment", nargs="+", type=str, default=None,
                        help="하드웨어 실험. 예: --experiment ionq ibm ionq-baseline ibm-baseline")
    stage.add_argument("--all", action="store_true", help="generate → train (순차)")
    stage.add_argument("--status", action="store_true", help="프로젝트 상태 확인")

    # 공통 옵션
    common = parser.add_argument_group("Common Options")
    common.add_argument("-c", "--code", nargs="+", type=str, default=None)
    common.add_argument("-n", "--noise", nargs="+", type=str, default=None)
    common.add_argument("-d", "--distance", nargs="+", type=int, default=None)
    common.add_argument("-e", "--error_type", nargs="+", type=str, default=None)
    common.add_argument("-m", "--models", nargs="+", type=str, default=None,
                        help="모델 선택 (학습 및 실험에 적용)")
    common.add_argument("--background", action="store_true", help="백그라운드 실행")

    # 데이터셋 생성 옵션
    generate = parser.add_argument_group("Generation Options")
    generate.add_argument("--cpu", nargs="?", type=str, default=None, const=None, help="num_workers 조정 (generate 단계)")
    
    # 학습 옵션
    train = parser.add_argument_group("Training Options")
    train.add_argument("-g", "--gpu", type=int, default=None, help="단일 GPU 번호")
    train.add_argument("--gpu0", nargs="+", type=str, default=None,
                       help="GPU 0 모델. 예: --gpu0 CNN GCN GAT APPNP")
    train.add_argument("--gpu1", nargs="+", type=str, default=None,
                       help="GPU 1 모델. 예: --gpu1 GCNII GNN GraphTransformer GraphMamba")

    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.generate, args.train, args.experiment, args.all, args.status]):
        print("사용법: python3 run_suite.py --help")
        return

    print(f"{'='*60}")
    print(f"  🔧 KCS Workflow Manager")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start = time.time()

    if args.status:
        do_status(args)
        return

    if args.all:
        args.generate = True
        args.train = True

    if args.generate:
        do_generate(args)

    if args.train:
        do_train(args)

    if args.experiment:
        do_experiment(args)

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"  ✨ Done! ({elapsed:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()