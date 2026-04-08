"""
Color Code용 커스텀 Stim 회로 및 데이터셋 생성기

Stim의 기본 color_code:memory_xyz는 superdense 방식(ancilla = face 수)을 사용하지만,
Qiskit/IonQ는 face × 2 (X-type + Z-type)를 개별 측정합니다.

이 파일은 Qiskit과 동일한 구조의 Stim 회로를 직접 작성하여,
학습 데이터와 하드웨어 데이터의 representation이 정확히 일치하도록 합니다.

지원: d=3 (7 data, 6 ancilla), d=5 (19 data, 18 ancilla), d=7 (37 data, 36 ancilla)
"""

import stim
import numpy as np
from typing import Tuple, List, Dict

# ==============================================================================
# Distance별 Face 정의 (Stim에서 추출)
# 각 face는 X-type + Z-type stabilizer 한 쌍으로 매핑됩니다.
# ==============================================================================
COLORCODE_FACES = {
    3: [
        [1, 2, 3, 4],
        [0, 1, 3, 5],
        [3, 4, 5, 6],
    ],
    5: [
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [0, 1, 5, 9],
        [2, 3, 6, 7, 10, 11],
        [5, 6, 9, 10, 12, 13],
        [7, 8, 11, 14],
        [10, 11, 13, 14, 15, 16],
        [12, 13, 15, 17],
        [15, 16, 17, 18],
    ],
    7: [
        [1, 2, 7, 8],
        [3, 4, 9, 10],
        [5, 6, 11, 12],
        [0, 1, 7, 13],
        [2, 3, 8, 9, 14, 15],
        [4, 5, 10, 11, 16, 17],
        [7, 8, 13, 14, 18, 19],
        [9, 10, 15, 16, 20, 21],
        [11, 12, 17, 22],
        [14, 15, 19, 20, 23, 24],
        [16, 17, 21, 22, 25, 26],
        [18, 19, 23, 27],
        [20, 21, 24, 25, 28, 29],
        [23, 24, 27, 28, 30, 31],
        [25, 26, 29, 32],
        [28, 29, 31, 32, 33, 34],
        [30, 31, 33, 35],
        [33, 34, 35, 36],
    ],
}

COLORCODE_DATA_COORDS = {
    3: {
        0: (0.0, 0.0), 1: (1.0, 0.0), 2: (3.0, 0.0),
        3: (1.5, 1.0), 4: (2.5, 1.0), 5: (1.0, 2.0),
        6: (1.5, 3.0),
    },
    5: {
        0: (0.0, 0.0), 1: (1.0, 0.0), 2: (3.0, 0.0), 3: (4.0, 0.0), 4: (6.0, 0.0),
        5: (1.5, 1.0), 6: (2.5, 1.0), 7: (4.5, 1.0), 8: (5.5, 1.0),
        9: (1.0, 2.0), 10: (3.0, 2.0), 11: (4.0, 2.0),
        12: (1.5, 3.0), 13: (2.5, 3.0), 14: (4.5, 3.0),
        15: (3.0, 4.0), 16: (4.0, 4.0),
        17: (2.5, 5.0),
        18: (3.0, 6.0),
    },
    7: {
        0: (0.0, 0.0), 1: (1.0, 0.0), 2: (3.0, 0.0), 3: (4.0, 0.0),
        4: (6.0, 0.0), 5: (7.0, 0.0), 6: (9.0, 0.0),
        7: (1.5, 1.0), 8: (2.5, 1.0), 9: (4.5, 1.0), 10: (5.5, 1.0),
        11: (7.5, 1.0), 12: (8.5, 1.0),
        13: (1.0, 2.0), 14: (3.0, 2.0), 15: (4.0, 2.0), 16: (6.0, 2.0), 17: (7.0, 2.0),
        18: (1.5, 3.0), 19: (2.5, 3.0), 20: (4.5, 3.0), 21: (5.5, 3.0), 22: (7.5, 3.0),
        23: (3.0, 4.0), 24: (4.0, 4.0), 25: (6.0, 4.0), 26: (7.0, 4.0),
        27: (2.5, 5.0), 28: (4.5, 5.0), 29: (5.5, 5.0),
        30: (3.0, 6.0), 31: (4.0, 6.0), 32: (6.0, 6.0),
        33: (4.5, 7.0), 34: (5.5, 7.0),
        35: (4.0, 8.0),
        36: (4.5, 9.0),
    },
}

# Data qubit 수
COLORCODE_NUM_DATA = {3: 7, 5: 19, 7: 37}

# Face 색상 (R=0.0, G=0.5, B=1.0)
# d=3: P0=R, P1=G, P2=B
# d=5: R={P0,P1,P6}, G={P2,P3,P7}, B={P4,P5,P8}
# d=7: 동일 패턴으로 확장 (Stim에서 추출)
COLORCODE_FACE_COLORS = {
    3: [0.0, 0.5, 1.0],           # R, G, B
    5: [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.5, 1.0],
    7: [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 1.0,
        0.5, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0],
}


def _get_code_config(distance: int) -> dict:
    """Distance에 따른 코드 구성 정보를 반환합니다."""
    if distance not in COLORCODE_FACES:
        raise ValueError(f"Color code distance {distance} not supported. Use 3, 5, or 7.")

    faces = COLORCODE_FACES[distance]
    num_data = COLORCODE_NUM_DATA[distance]
    num_faces = len(faces)
    num_ancilla = num_faces * 2  # X-type + Z-type

    data_qubits = list(range(num_data))
    x_ancilla = list(range(num_data, num_data + num_faces))
    z_ancilla = list(range(num_data + num_faces, num_data + num_ancilla))

    x_stabilizers = {x_ancilla[i]: faces[i] for i in range(num_faces)}
    z_stabilizers = {z_ancilla[i]: faces[i] for i in range(num_faces)}

    return {
        "num_data": num_data,
        "num_faces": num_faces,
        "num_ancilla": num_ancilla,
        "data_qubits": data_qubits,
        "x_ancilla": x_ancilla,
        "z_ancilla": z_ancilla,
        "x_stabilizers": x_stabilizers,
        "z_stabilizers": z_stabilizers,
    }


def create_color_code_circuit(distance: int, rounds: int, noise: float,
                               meas_noise: float = 0.0,
                               reset_noise: float = 0.0,
                               gate_noise: float = 0.0) -> stim.Circuit:
    """호환 래퍼: _build_stim_circuit을 호출합니다."""
    return _build_stim_circuit(distance, rounds,
                                data_depol=noise,
                                meas_noise=meas_noise,
                                reset_noise=reset_noise,
                                gate_noise=gate_noise)


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
    cfg = _get_code_config(distance)
    data_qubits = cfg["data_qubits"]
    x_ancilla = cfg["x_ancilla"]
    z_ancilla = cfg["z_ancilla"]
    all_ancilla = x_ancilla + z_ancilla
    x_stabs = cfg["x_stabilizers"]
    z_stabs = cfg["z_stabilizers"]

    circuit = stim.Circuit()

    # 초기 Reset
    circuit.append("R", data_qubits + all_ancilla)
    if reset_noise > 0:
        circuit.append("X_ERROR", data_qubits + all_ancilla, reset_noise)

    for r in range(rounds):
        if data_depol > 0:
            circuit.append("DEPOLARIZE1", data_qubits, data_depol)

        # Reset ancilla
        circuit.append("R", all_ancilla)
        if reset_noise > 0:
            circuit.append("X_ERROR", all_ancilla, reset_noise)

        # X-type syndrome
        circuit.append("H", x_ancilla)
        if gate_noise > 0:
            circuit.append("DEPOLARIZE1", x_ancilla, gate_noise)

        for anc, data_list in x_stabs.items():
            for d in data_list:
                circuit.append("CX", [anc, d])
                if gate_noise > 0:
                    circuit.append("DEPOLARIZE2", [anc, d], gate_noise)

        circuit.append("H", x_ancilla)
        if gate_noise > 0:
            circuit.append("DEPOLARIZE1", x_ancilla, gate_noise)

        if meas_noise > 0:
            circuit.append("X_ERROR", x_ancilla, meas_noise)
        circuit.append("MR", x_ancilla)

        # Z-type syndrome
        for anc, data_list in z_stabs.items():
            for d in data_list:
                circuit.append("CX", [d, anc])
                if gate_noise > 0:
                    circuit.append("DEPOLARIZE2", [d, anc], gate_noise)

        if meas_noise > 0:
            circuit.append("X_ERROR", z_ancilla, meas_noise)
        circuit.append("MR", z_ancilla)

    # 최종 data qubit 측정
    if meas_noise > 0:
        circuit.append("X_ERROR", data_qubits, meas_noise)
    circuit.append("M", data_qubits)

    return circuit


def _extract_data_qubit_indices(circuit: stim.Circuit = None, distance: int = 3) -> List[int]:
    """Color code의 data qubit 인덱스를 반환합니다."""
    return list(range(COLORCODE_NUM_DATA[distance]))


def generate_dataset(
    distance: int, rounds: int, noise_rate: float, shots: int,
    error_type: str = "X",
    meas_noise: float = 0.0, reset_noise: float = 0.0, gate_noise: float = 0.0,
    data_depol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Color Code용 (신드롬, 물리적 에러) 데이터셋을 생성합니다.

    커스텀 Stim 회로를 사용하여 Qiskit/IonQ와 동일한 구조로 데이터를 생성합니다.
    """
    cfg = _get_code_config(distance)
    data_indices = cfg["data_qubits"]
    num_data = cfg["num_data"]
    num_ancilla = cfg["num_ancilla"]
    num_detectors = num_ancilla * rounds

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
        error_mask = np.random.random(num_data) < noise_rate
        physical_errors[i] = error_mask.astype(np.float32)

        error_stim_indices = [data_indices[j] for j in range(num_data) if error_mask[j]]

        noisy_circuit = stim.Circuit()
        injected = False
        for instruction in flat_circuit:
            noisy_circuit.append(instruction)
            if not injected and instruction.name in ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]:
                if len(error_stim_indices) > 0:
                    noisy_circuit.append(stim_error_cmd, error_stim_indices, 1.0)
                injected = True

        sampler = noisy_circuit.compile_sampler()
        raw_meas = sampler.sample(shots=1)[0]

        ancilla_meas = raw_meas[:num_ancilla * rounds].reshape(rounds, num_ancilla)

        detectors = np.zeros((rounds, num_ancilla), dtype=np.float32)
        detectors[0] = ancilla_meas[0].astype(np.float32)
        for r in range(1, rounds):
            detectors[r] = (ancilla_meas[r] != ancilla_meas[r - 1]).astype(np.float32)

        detector_data[i] = detectors.flatten()

    return detector_data, physical_errors