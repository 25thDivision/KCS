import stim
import numpy as np
from typing import Tuple

def create_color_code_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
    """
    2D Color Code를 위한 Stim 회로를 생성합니다
    (참고: 이 함수는 main.py와의 호환성을 위해 유지하거나, 필요 없다면 삭제해도 됩니다)
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
    물리적 에러 위치(Label)와 신드롬(Input)을 포함한 데이터셋을 생성합니다
    Stim의 TableauSimulator를 사용하여 Code Capacity Noise Model을 적용합니다
    
    Args:
        distance (int): 코드 거리입니다
        rounds (int): 측정 라운드 수입니다 (Code Capacity 모델에서는 보통 무시되거나 1로 설정됩니다)
        noise_rate (float): 물리적 에러 확률(p)입니다
        shots (int): 생성할 데이터의 개수입니다
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - detector_data: [shots, num_detectors] 크기의 신드롬 데이터입니다
            - physical_errors: [shots, num_qubits] 크기의 물리적 에러 라벨입니다 (0: 정상, 1: 에러)
    """
    # 1. 회로 구조를 생성합니다 (노이즈는 시뮬레이터에서 직접 주입하므로 여기선 0으로 설정합니다)
    circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=0
    )
    
    num_detectors = circuit.num_detectors
    num_qubits = circuit.num_qubits
    
    # 2. 데이터 저장소를 초기화합니다
    detector_data = np.zeros((shots, num_detectors), dtype=np.bool_)
    physical_errors = np.zeros((shots, num_qubits), dtype=np.uint8)
    
    print(f"[Generator] Generating {shots} samples with physical error labels...")

    # 3. 시뮬레이터를 초기화합니다
    sim = stim.TableauSimulator()

    for i in range(shots):
        # 3-1. 회로를 실행하여 상태를 초기화합니다
        sim.do(circuit)
        
        # 3-2. 랜덤한 물리적 에러 벡터를 생성합니다 (Code Capacity Noise Model)
        # 각 큐비트에 대해 noise_rate 확률로 에러(여기선 Z 에러 가정)를 발생시킵니다
        err_vector = np.random.binomial(1, noise_rate, size=num_qubits).astype(np.uint8)
        physical_errors[i] = err_vector
        
        # 에러가 발생한 큐비트의 인덱스를 추출합니다
        error_indices = np.flatnonzero(err_vector)
        
        # 3-3. 시뮬레이터에 에러를 주입합니다
        # Color Code에서 Z 에러는 X Stabilizer를 트리거하므로 Z_ERROR 연산을 사용합니다
        if len(error_indices) > 0:
            # stim 명령어 포맷에 맞춰 문자열을 구성합니다
            targets = " ".join(map(str, error_indices))
            sim.do(stim.Circuit(f"Z_ERROR(1) {targets}"))
            
        # 3-4. 측정 결과로부터 신드롬(Detector event)을 계산합니다
        # Code Capacity 모델(Single Round)에서는 측정값 처리가 비교적 단순합니다
        # 정확한 라벨링을 위해 에러가 주입된 임시 회로를 만들어 컴파일러를 사용합니다
        
        noisy_circuit = circuit.copy()
        if len(error_indices) > 0:
             noisy_circuit.append("Z_ERROR", error_indices, 1.0) # 100% 확률로 에러 주입
        
        # 이 샷에 대한 신드롬을 추출합니다
        sampler = noisy_circuit.compile_detector_sampler()
        shot_result = sampler.sample(shots=1, bit_packed=False)[0]
        detector_data[i] = shot_result

    print("[Generator] Data generation complete.")
    return detector_data, physical_errors