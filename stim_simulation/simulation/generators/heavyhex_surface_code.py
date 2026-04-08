"""
Heavy-Hex Surface Code d=3 — Stim 회로 및 데이터셋 생성기

TRUE stabilizer 기반 구현 (GF(2) 선형대수학으로 도출):
  Z stabilizers: Z{0,1,3,4}, Z{1,2,4,5}, Z{0,1}, Z{7,8}
  X stabilizers: X{0,1,2,6}, X{2,3,4,6}, X{2,5}, X{7,8}
  Logical Z = Z{0,3,6}
  Logical X = X{0,1,2}

하드웨어의 fold/unfold depth-7 회로는 비가환 연산자를 측정하지만,
TRUE stabilizer는 모두 상호 가환하므로 Stim에서는 표준 ancilla 기반
순차 측정이 가능합니다.

큐빗 배치:
  0-8:   data qubits (9개)
  9-12:  Z-type ancilla (4개)
  13-16: X-type ancilla (4개)

Data qubit 매핑 (Stim → ibm_boston):
  0→43, 1→45, 2→47, 3→63, 4→65, 5→67, 6→83, 7→85, 8→87

Grid layout (3x3):
  D0(43)  D1(45)  D2(47)
  D3(63)  D4(65)  D5(67)
  D6(83)  D7(85)  D8(87)

측정 순서 (per round, 8 bits):
  [Z{0134}, Z{1245}, Z{01}, Z{78}, X{0126}, X{2346}, X{25}, X{78}]
"""

import stim
import numpy as np
from typing import Tuple, List

# ==============================================================================
# Stabilizer 정의 (distance=3)
# ==============================================================================
HEAVYHEX_D3 = {
    "num_data": 9,
    "num_ancilla": 8,
    "data_qubits": list(range(9)),
    "z_ancilla": [9, 10, 11, 12],
    "x_ancilla": [13, 14, 15, 16],
    "z_stabilizers": {
        9:  [0, 1, 3, 4],  # Z1 (bulk, weight-4)
        10: [1, 2, 4, 5],  # Z2 (bulk, weight-4)
        11: [0, 1],        # Zb_top (boundary, weight-2)
        12: [7, 8],        # Zb_bot (boundary, weight-2)
    },
    "x_stabilizers": {
        13: [0, 1, 2, 6],  # X_A (bulk, weight-4)
        14: [2, 3, 4, 6],  # X_B (bulk, weight-4)
        15: [2, 5],        # Xb_1 (boundary, weight-2)
        16: [7, 8],        # Xb_2 (boundary, weight-2)
    },
    "logical_z": [0, 3, 6],
    "logical_x": [0, 1, 2],
    "hw_mapping": {
        0: 43, 1: 45, 2: 47,
        3: 63, 4: 65, 5: 67,
        6: 83, 7: 85, 8: 87,
    },
}


def _build_stim_circuit(distance: int, rounds: int,
                        data_depol: float = 0.0,
                        meas_noise: float = 0.0,
                        reset_noise: float = 0.0,
                        gate_noise: float = 0.0) -> stim.Circuit:
    """
    Heavy-hex surface code d=3의 TRUE stabilizer 기반 Stim 회로를 생성합니다.

    측정 순서 (per round):
      1. Z-type syndrome: CX(data->anc), MR(anc) -> X 에러 감지
      2. X-type syndrome: H, CX(anc->data), H, MR(anc) -> Z 에러 감지

    마지막에 data qubit 전체를 측정합니다.
    """
    cfg = HEAVYHEX_D3
    data_qubits = cfg["data_qubits"]
    z_ancilla = cfg["z_ancilla"]
    x_ancilla = cfg["x_ancilla"]
    all_ancilla = z_ancilla + x_ancilla
    z_stabs = cfg["z_stabilizers"]
    x_stabs = cfg["x_stabilizers"]

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

        # === Z-type syndrome extraction ===
        # CX: data (control) -> Z-ancilla (target) -> Z-parity 측정 -> X 에러 감지
        for anc, data_list in z_stabs.items():
            for d in data_list:
                circuit.append("CX", [d, anc])
                if gate_noise > 0:
                    circuit.append("DEPOLARIZE2", [d, anc], gate_noise)

        if meas_noise > 0:
            circuit.append("X_ERROR", z_ancilla, meas_noise)
        circuit.append("MR", z_ancilla)

        # === X-type syndrome extraction ===
        circuit.append("H", x_ancilla)
        if gate_noise > 0:
            circuit.append("DEPOLARIZE1", x_ancilla, gate_noise)

        # CX: X-ancilla (control) -> data (target) -> X-parity 측정 -> Z 에러 감지
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

    # 최종 data qubit 측정
    if meas_noise > 0:
        circuit.append("X_ERROR", data_qubits, meas_noise)
    circuit.append("M", data_qubits)

    return circuit


def create_heavyhex_surface_code_circuit(
    distance: int, rounds: int, noise: float,
    meas_noise: float = 0.0,
    reset_noise: float = 0.0,
    gate_noise: float = 0.0
) -> stim.Circuit:
    """Pipeline 호환 인터페이스: heavy-hex surface code Stim 회로 생성."""
    if distance != 3:
        raise NotImplementedError(
            f"Heavy-hex surface code currently only supports d=3, got d={distance}"
        )
    return _build_stim_circuit(
        distance, rounds,
        data_depol=noise,
        meas_noise=meas_noise,
        reset_noise=reset_noise,
        gate_noise=gate_noise,
    )


def _extract_data_qubit_indices(circuit: stim.Circuit = None) -> List[int]:
    """Heavy-hex surface code d=3의 data qubit 인덱스를 반환합니다."""
    return list(range(9))


def _extract_ancilla_qubit_indices() -> List[int]:
    """Heavy-hex surface code d=3의 ancilla qubit 인덱스를 반환합니다."""
    return list(range(9, 17))


def generate_dataset(
    distance: int, rounds: int, noise_rate: float, shots: int,
    error_type: str = "X",
    meas_noise: float = 0.0, reset_noise: float = 0.0, gate_noise: float = 0.0,
    data_depol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heavy-Hex Surface Code용 (신드롬, 물리적 에러) 데이터셋을 생성합니다.

    color_code.py / surface_code.py와 동일한 API:
      1. Clean circuit 생성 (background noise만)
      2. shot별로 data qubit에 에러 주입
      3. compile_sampler()로 raw measurement 추출
      4. Ancilla measurement에 temporal differencing 적용
      5. (syndromes, physical_errors) 반환

    Returns:
        (detector_data, physical_errors):
          detector_data: shape (shots, rounds * 8), temporal-differenced syndromes
          physical_errors: shape (shots, 9), data qubit error labels
    """
    cfg = HEAVYHEX_D3
    data_indices = cfg["data_qubits"]
    num_data = len(data_indices)
    num_ancilla = cfg["num_ancilla"]
    num_detectors = num_ancilla * rounds

    clean_circuit = _build_stim_circuit(
        distance, rounds,
        data_depol=data_depol,
        meas_noise=meas_noise,
        reset_noise=reset_noise,
        gate_noise=gate_noise,
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


def get_stabilizer_info() -> dict:
    """
    Stabilizer 정보를 반환합니다 (graph mapper 등에서 사용).
    """
    cfg = HEAVYHEX_D3
    return {
        "x_stabilizers": [cfg["x_stabilizers"][a] for a in cfg["x_ancilla"]],
        "z_stabilizers": [cfg["z_stabilizers"][a] for a in cfg["z_ancilla"]],
        "logical_z": cfg["logical_z"],
        "logical_x": cfg["logical_x"],
        "hw_mapping": cfg["hw_mapping"],
        "num_data": cfg["num_data"],
        "num_ancilla": cfg["num_ancilla"],
    }

# =============================================================================
# Hardware ↔ Stim syndrome 순서 매핑 (IBM heavy-hex depth-7)
# =============================================================================
# HW 측정 순서 (per cycle, heavyhex_surface_code_depth7.py syn_bit 추적):
#   bit 0: Z1=Z{0,1,3,4}       bit 4: Z2=Z{1,2,4,5}
#   bit 1: X1=X{0,3,6,7}       bit 5: X2=X{2,5,7,8}
#   bit 2: Xb_right=X{2,5}     bit 6: Xb_left=X{0,3}
#   bit 3: Zb_top=Z{0,1}       bit 7: Zb_bot=Z{7,8}
#
# Stim 측정 순서 (per round):
#   bit 0: Z{0,1,3,4}  bit 4: X{0,1,2,6}
#   bit 1: Z{1,2,4,5}  bit 5: X{2,3,4,6}
#   bit 2: Z{0,1}      bit 6: X{2,5}
#   bit 3: Z{7,8}      bit 7: X{7,8}

# Z stab: HW와 Stim이 동일한 연산자 (X 에러 학습용)
HW_Z_TO_STIM = {
    0: 0,   # HW Z1=Z{0,1,3,4} → Stim bit 0
    3: 2,   # HW Zb_top=Z{0,1}  → Stim bit 2
    4: 1,   # HW Z2=Z{1,2,4,5}  → Stim bit 1
    7: 3,   # HW Zb_bot=Z{7,8}  → Stim bit 3
}

# X stab: HW와 Stim이 다른 연산자이므로 직접 매핑 불가
# HW bit 2 (X{2,5}) = Stim bit 6 (exact match)
# HW bit 5 (X{2,5,7,8}) = Stim bit 6 XOR Stim bit 7 (product)
# HW bit 1 (X{0,3,6,7}) = NOT in stabilizer group
# HW bit 6 (X{0,3}) = NOT in stabilizer group


def reorder_hw_to_stim(hw_syndromes: np.ndarray, num_rounds: int) -> np.ndarray:
    """
    하드웨어 syndrome → Stim 학습 데이터 순서로 재배열 (X 에러 전용).

    Z stab 4비트만 정확히 매핑하고, X stab 4비트는 0으로 마스킹합니다.
    IBM experiment에서 raw syndrome 추출 직후, mapper 전에 호출:

        from generators.heavyhex_surface_code import reorder_hw_to_stim
        reordered = reorder_hw_to_stim(hw_syndrome, num_rounds)
        features = mapper.map_to_node_features(reordered)

    Args:
        hw_syndromes: shape (shots, num_rounds * 8), HW bit 순서
        num_rounds: QEC 라운드 수

    Returns:
        stim_syndromes: shape (shots, num_rounds * 8), Stim bit 순서, X stab=0
    """
    shots = hw_syndromes.shape[0]
    stim_syndromes = np.zeros_like(hw_syndromes)

    for r in range(num_rounds):
        hw_base = r * 8
        stim_base = r * 8
        for hw_bit, stim_bit in HW_Z_TO_STIM.items():
            stim_syndromes[:, stim_base + stim_bit] = hw_syndromes[:, hw_base + hw_bit]
        # X stab bits (stim 4-7) remain 0

    return stim_syndromes