"""
Color Code용 커스텀 Stim 회로 및 데이터셋 생성기

Stim의 기본 color_code:memory_xyz는 superdense 방식(3 ancilla/round)을 사용하지만,
Qiskit/IonQ는 6 ancilla/round (3 X-type + 3 Z-type)를 개별 측정합니다.

이 파일은 Qiskit과 동일한 6-ancilla 구조의 Stim 회로를 직접 작성하여,
학습 데이터와 하드웨어 데이터의 representation이 정확히 일치하도록 합니다.

큐빗 배치:
  0-6:   data qubits (7개)
  7-9:   X-type ancilla (R, G, B)
  10-12: Z-type ancilla (R, G, B)

Stabilizer 연결:
  X/Z-Red:   data [0, 1, 2, 3]
  X/Z-Green: data [1, 2, 4, 5]
  X/Z-Blue:  data [2, 3, 5, 6]
"""

import stim
import numpy as np
from typing import Tuple, List

# ==============================================================================
# Stabilizer 정의 (distance=3 기준, 추후 동적 확장 가능)
# ==============================================================================
COLORCODE_D3 = {
    "num_data": 7,
    "num_ancilla": 6,
    "data_qubits": list(range(7)),          # 0-6
    "x_ancilla": [7, 8, 9],                 # X-type R, G, B
    "z_ancilla": [10, 11, 12],              # Z-type R, G, B
    "x_stabilizers": {
        7: [0, 1, 2, 3],   # X-Red
        8: [1, 2, 4, 5],   # X-Green
        9: [2, 3, 5, 6],   # X-Blue
    },
    "z_stabilizers": {
        10: [0, 1, 2, 3],  # Z-Red
        11: [1, 2, 4, 5],  # Z-Green
        12: [2, 3, 5, 6],  # Z-Blue
    },
}


def _build_stim_circuit(distance: int, rounds: int,
                        data_depol: float = 0.0,
                        meas_noise: float = 0.0,
                        reset_noise: float = 0.0,
                        gate_noise: float = 0.0) -> stim.Circuit:
    """
    Qiskit color code와 동일한 구조의 Stim 회로를 생성합니다.

    각 라운드:
      1. Reset ancilla
      2. X-type syndrome: H - CX(ancilla→data) - H - MR
      3. Z-type syndrome: CX(data→ancilla) - MR

    마지막에 data qubit 전체를 측정합니다.
    """
    cfg = COLORCODE_D3
    data_qubits = cfg["data_qubits"]
    x_ancilla = cfg["x_ancilla"]
    z_ancilla = cfg["z_ancilla"]
    all_ancilla = x_ancilla + z_ancilla
    x_stabs = cfg["x_stabilizers"]
    z_stabs = cfg["z_stabilizers"]

    circuit = stim.Circuit()

    # 초기 Reset (모든 큐빗)
    circuit.append("R", data_qubits + all_ancilla)
    if reset_noise > 0:
        circuit.append("X_ERROR", data_qubits + all_ancilla, reset_noise)

    for r in range(rounds):
        # Data qubit depolarization (라운드 시작 시)
        if data_depol > 0:
            circuit.append("DEPOLARIZE1", data_qubits, data_depol)

        # Reset ancilla
        circuit.append("R", all_ancilla)
        if reset_noise > 0:
            circuit.append("X_ERROR", all_ancilla, reset_noise)

        # === X-type syndrome extraction ===
        # H on X-ancilla
        circuit.append("H", x_ancilla)
        if gate_noise > 0:
            circuit.append("DEPOLARIZE1", x_ancilla, gate_noise)

        # CX: X-ancilla (control) → data (target)
        for anc, data_list in x_stabs.items():
            for d in data_list:
                circuit.append("CX", [anc, d])
                if gate_noise > 0:
                    circuit.append("DEPOLARIZE2", [anc, d], gate_noise)

        # H on X-ancilla
        circuit.append("H", x_ancilla)
        if gate_noise > 0:
            circuit.append("DEPOLARIZE1", x_ancilla, gate_noise)

        # Measure + Reset X-ancilla
        if meas_noise > 0:
            circuit.append("X_ERROR", x_ancilla, meas_noise)
        circuit.append("MR", x_ancilla)

        # === Z-type syndrome extraction ===
        # CX: data (control) → Z-ancilla (target)
        for anc, data_list in z_stabs.items():
            for d in data_list:
                circuit.append("CX", [d, anc])
                if gate_noise > 0:
                    circuit.append("DEPOLARIZE2", [d, anc], gate_noise)

        # Measure + Reset Z-ancilla
        if meas_noise > 0:
            circuit.append("X_ERROR", z_ancilla, meas_noise)
        circuit.append("MR", z_ancilla)

    # 최종 data qubit 측정
    if meas_noise > 0:
        circuit.append("X_ERROR", data_qubits, meas_noise)
    circuit.append("M", data_qubits)

    return circuit


def _extract_data_qubit_indices(circuit: stim.Circuit = None) -> List[int]:
    """Color code d=3의 data qubit 인덱스를 반환합니다."""
    return list(range(7))


def _extract_ancilla_qubit_indices() -> List[int]:
    """Color code d=3의 ancilla qubit 인덱스를 반환합니다."""
    return list(range(7, 13))


def generate_dataset(
    distance: int, rounds: int, noise_rate: float, shots: int,
    error_type: str = "X",
    meas_noise: float = 0.0, reset_noise: float = 0.0, gate_noise: float = 0.0,
    data_depol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Color Code용 (신드롬, 물리적 에러) 데이터셋을 생성합니다.

    커스텀 Stim 회로를 사용하여 Qiskit/IonQ와 동일한 6-ancilla 구조로
    데이터를 생성합니다.

    compile_sampler()로 raw measurement를 추출한 뒤, ancilla 측정값에
    수동 temporal differencing을 적용합니다.

    에러는 data qubit에만 주입되며, label도 data qubit만 포함합니다.
    """
    cfg = COLORCODE_D3
    data_indices = cfg["data_qubits"]
    num_data = len(data_indices)
    num_ancilla = cfg["num_ancilla"]  # 6
    num_detectors = num_ancilla * rounds  # 6 * rounds = 18 (d=3, rounds=3)

    # Stim 회로 생성
    clean_circuit = _build_stim_circuit(
        distance, rounds,
        data_depol=data_depol,
        meas_noise=meas_noise,
        reset_noise=reset_noise,
        gate_noise=gate_noise
    )
    flat_circuit = clean_circuit.flattened()

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

        # compile_sampler()로 raw measurement 추출
        sampler = noisy_circuit.compile_sampler()
        raw_meas = sampler.sample(shots=1)[0]

        # ancilla measurement 추출
        # 측정 순서: [round0_x0, round0_x1, round0_x2, round0_z0, round0_z1, round0_z2,
        #            round1_x0, ..., roundR_z2, data0, ..., data6]
        ancilla_meas = raw_meas[:num_ancilla * rounds].reshape(rounds, num_ancilla)

        # temporal differencing
        detectors = np.zeros((rounds, num_ancilla), dtype=np.float32)
        detectors[0] = ancilla_meas[0].astype(np.float32)
        for r in range(1, rounds):
            detectors[r] = (ancilla_meas[r] != ancilla_meas[r - 1]).astype(np.float32)

        detector_data[i] = detectors.flatten()

    return detector_data, physical_errors