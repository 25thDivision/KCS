import stim
import numpy as np
from typing import Tuple

def create_color_code_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
    """
    Main.py 등에서 회로 구조를 가져오기 위한 함수
    """
    return stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise
    )

def generate_dataset(
    distance: int, 
    rounds: int, 
    noise_rate: float, 
    shots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stim 회로의 초기화(Reset) 단계 직후에 에러를 강제로 주입하여
    (물리적 에러 위치, 신드롬) 데이터셋을 생성합니다.
    """
    # 1. 기준 회로 생성
    clean_circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=0
    )
    
    # [핵심 수정] REPEAT 블록을 풀어서 모든 명령어를 일렬로 나열합니다.
    # 그래야 숨어있는 Reset 명령어를 찾을 수 있습니다.
    flat_circuit = clean_circuit.flattened()
    
    num_qubits = clean_circuit.num_qubits
    num_detectors = clean_circuit.num_detectors
    
    detector_data = np.zeros((shots, num_detectors), dtype=np.float32)
    physical_errors = np.zeros((shots, num_qubits), dtype=np.float32)
    
    print(f"[Generator] Sampling {shots} shots with noise p={noise_rate} (Flattened Splicing)...")

    for i in range(shots):
        # 2. 에러 벡터 생성
        error_mask = np.random.random(num_qubits) < noise_rate
        physical_errors[i] = error_mask.astype(np.float32)
        error_indices = np.where(error_mask)[0]
        
        # 3. 노이즈 주입 회로 구성
        noisy_circuit = stim.Circuit()
        injected = False
        
        # flat_circuit을 순회하면 숨어있던 명령어들이 다 보입니다.
        for instruction in flat_circuit:
            noisy_circuit.append(instruction)
            
            # Reset(초기화) 직후 에러 주입
            # (MR, R, RX 등 모든 초기화 관련 명령어 감지)
            if not injected and instruction.name in ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]:
                if len(error_indices) > 0:
                    noisy_circuit.append("X_ERROR", error_indices, 1.0)
                    # [디버그] 첫 번째 샷에서만 주입 로그 출력
                    if i == 0:
                        print(f"  [Debug] Shot #0: Injected Z errors on qubits {error_indices[:5]}...")
                injected = True
        
        # 4. 샘플링
        sampler = noisy_circuit.compile_detector_sampler()
        shot_result = sampler.sample(shots=1)[0]
        detector_data[i] = shot_result.astype(np.float32)

    print("[Generator] Data generation complete.")
    return detector_data, physical_errors