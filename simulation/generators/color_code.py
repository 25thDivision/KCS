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
    shots: int,
    error_type: str = "X"  # [추가] 에러 종류를 선택받습니다 (기본값 X)
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
    
    # REPEAT 블록 풀기
    flat_circuit = clean_circuit.flattened()
    
    num_qubits = clean_circuit.num_qubits
    num_detectors = clean_circuit.num_detectors
    
    detector_data = np.zeros((shots, num_detectors), dtype=np.float32)
    physical_errors = np.zeros((shots, num_qubits), dtype=np.float32)
    
    # [설정] 주입할 에러 명령어 결정
    stim_error_cmd = "X_ERROR" if error_type == "X" else "Z_ERROR"
    
    # 배치 처리는 외부 스크립트(generate_dataset_*.py)에서 하므로,
    # 여기서는 요청받은 shots 만큼만 생성하면 됩니다.
    for i in range(shots):
        # 2. 에러 벡터 생성
        error_mask = np.random.random(num_qubits) < noise_rate
        physical_errors[i] = error_mask.astype(np.float32)
        error_indices = np.where(error_mask)[0]
        
        # 3. 노이즈 주입 회로 구성
        noisy_circuit = stim.Circuit()
        injected = False
        
        for instruction in flat_circuit:
            noisy_circuit.append(instruction)
            
            # Reset 직후 에러 주입
            if not injected and instruction.name in ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]:
                if len(error_indices) > 0:
                    # [수정] 선택된 에러 타입(X 또는 Z)을 주입
                    noisy_circuit.append(stim_error_cmd, error_indices, 1.0)
                injected = True
        
        # 4. 샘플링
        sampler = noisy_circuit.compile_detector_sampler()
        shot_result = sampler.sample(shots=1)[0]
        detector_data[i] = shot_result.astype(np.float32)

    return detector_data, physical_errors