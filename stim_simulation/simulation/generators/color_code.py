import stim
import numpy as np
from typing import Tuple, List


def create_color_code_circuit(distance: int, rounds: int, noise: float,
                               meas_noise: float = 0.0,
                               reset_noise: float = 0.0,
                               gate_noise: float = 0.0) -> stim.Circuit:
    return stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise,
        before_measure_flip_probability=meas_noise,
        after_reset_flip_probability=reset_noise,
        after_clifford_depolarization=gate_noise,
    )


def _extract_data_qubit_indices(circuit: stim.Circuit) -> List[int]:
    """
    Stim 회로에서 data qubit 인덱스를 추출합니다.

    before_round_data_depolarization > 0 으로 생성된 회로에서
    DEPOLARIZE1이 적용되는 큐빗 = data qubit입니다.
    """
    data_indices = set()
    for inst in circuit.flattened():
        if inst.name == "DEPOLARIZE1":
            for t in inst.targets_copy():
                data_indices.add(t.value)
    return sorted(data_indices)


def generate_dataset(
    distance: int, rounds: int, noise_rate: float, shots: int,
    error_type: str = "X",
    data_depol: float = 0.0, meas_noise: float = 0.0, reset_noise: float = 0.0, gate_noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Color Code용 (신드롬, 물리적 에러) 데이터셋을 생성합니다.

    에러는 data qubit에만 주입되며, label도 data qubit만 포함합니다.
    (d=3: 7 data qubits → label 7차원)
    """
    # data qubit 인덱스 추출용 참조 회로 (dp > 0 필요)
    ref_circuit = create_color_code_circuit(distance, rounds, 0.001,
                                            meas_noise=0,
                                            reset_noise=0,
                                            gate_noise=0)
    data_indices = _extract_data_qubit_indices(ref_circuit)
    num_data = len(data_indices)

    # 실제 시뮬레이션용 회로 (dp=0, 에러는 수동 주입)
    clean_circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance, rounds=rounds,
        before_round_data_depolarization=data_depol,
        before_measure_flip_probability=meas_noise,
        after_reset_flip_probability=reset_noise,
        after_clifford_depolarization=gate_noise,
    )
    flat_circuit = clean_circuit.flattened()
    num_detectors = clean_circuit.num_detectors

    detector_data = np.zeros((shots, num_detectors), dtype=np.float32)
    physical_errors = np.zeros((shots, num_data), dtype=np.float32)
    stim_error_cmd = "X_ERROR" if error_type == "X" else "Z_ERROR"

    for i in range(shots):
        # data qubit에만 에러 생성
        error_mask = np.random.random(num_data) < noise_rate
        physical_errors[i] = error_mask.astype(np.float32)

        # 에러가 있는 data qubit의 Stim 인덱스
        error_stim_indices = [data_indices[j] for j in range(num_data) if error_mask[j]]

        noisy_circuit = stim.Circuit()
        injected = False
        for instruction in flat_circuit:
            noisy_circuit.append(instruction)
            if not injected and instruction.name in ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]:
                if len(error_stim_indices) > 0:
                    noisy_circuit.append(stim_error_cmd, error_stim_indices, 1.0)
                injected = True

        sampler = noisy_circuit.compile_detector_sampler()
        shot_result = sampler.sample(shots=1)[0]
        detector_data[i] = shot_result.astype(np.float32)

    return detector_data, physical_errors