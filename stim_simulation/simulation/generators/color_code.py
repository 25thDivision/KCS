import stim
import numpy as np
from typing import Tuple

def create_color_code_circuit(distance: int, rounds: int, noise: float,
                               meas_noise: float = 0.0) -> stim.Circuit:
    """
    Color Code 회로를 생성합니다.
    
    Args:
        distance: Code distance
        rounds: QEC 라운드 수
        noise: 데이터 큐빗 depolarization 확률
        meas_noise: 측정 에러 확률 (0이면 기존과 동일)
    """
    return stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise,
        before_measure_flip_probability=meas_noise,
        after_reset_flip_probability=meas_noise,
    )

def generate_dataset(
    distance: int, 
    rounds: int, 
    noise_rate: float, 
    shots: int,
    error_type: str = "X",
    meas_noise: float = 0.005,  # ★ 추가: 측정 노이즈 (기본 0.5%)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stim 회로의 초기화(Reset) 단계 직후에 에러를 강제로 주입하여
    (물리적 에러 위치, 신드롬) 데이터셋을 생성합니다.
    
    ★ meas_noise > 0이면 측정 과정에도 노이즈가 추가되어,
      실제 하드웨어(IonQ, IBM 등)의 측정 에러를 모사합니다.
    
    Args:
        distance: Code distance
        rounds: QEC 라운드 수
        noise_rate: 데이터 큐빗 에러 확률
        shots: 생성할 샘플 수
        error_type: "X" 또는 "Z"
        meas_noise: 측정 에러 확률 (0이면 기존과 동일한 clean 측정)
    """
    # 1. 기준 회로 생성
    # ★ 핵심 변경: 측정 노이즈를 회로 템플릿에 포함
    clean_circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=0,
        before_measure_flip_probability=meas_noise,
        after_reset_flip_probability=meas_noise,
    )
    
    # REPEAT 블록 풀기
    flat_circuit = clean_circuit.flattened()
    
    num_qubits = clean_circuit.num_qubits
    num_detectors = clean_circuit.num_detectors
    
    detector_data = np.zeros((shots, num_detectors), dtype=np.float32)
    physical_errors = np.zeros((shots, num_qubits), dtype=np.float32)
    
    # [설정] 주입할 에러 명령어 결정
    stim_error_cmd = "X_ERROR" if error_type == "X" else "Z_ERROR"
    
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
                    noisy_circuit.append(stim_error_cmd, error_indices, 1.0)
                injected = True
        
        # 4. 샘플링
        # ★ 회로 템플릿에 측정 노이즈가 포함되어 있으므로,
        #   detector 결과에 자연스럽게 측정 에러가 반영됨
        sampler = noisy_circuit.compile_detector_sampler()
        shot_result = sampler.sample(shots=1)[0]
        detector_data[i] = shot_result.astype(np.float32)

    return detector_data, physical_errors