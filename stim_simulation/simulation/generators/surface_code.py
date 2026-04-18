"""
Rotated Surface Code Stim circuit + dataset generator.

The stabilizer convention here must match the hardware/Qiskit side so that ML
decoders trained on this output can be used on ibm_experiment syndromes.

We delegate circuit construction to
ibm_experiment/extractors/stim_compat.py::build_qiskit_style_stim_circuit,
which mirrors the schedule used by ibm_experiment SurfaceCodeCircuit:
  - X stabilizers on one bulk diagonal + left/right boundaries
  - Z stabilizers on the other bulk diagonal + top/bottom boundaries
  - Logical Z = left column of data grid
  - MR order per round: [X_0..X_{nx-1}, Z_0..Z_{nz-1}]
  - Data qubits indexed 0..d^2-1
"""

import os
import sys
import numpy as np
import stim
from typing import Tuple, List

# Ensure ibm_experiment and stim_simulation are on sys.path when this module
# is imported directly from a dataset-generation script.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_IBM = os.path.join(_ROOT, "ibm_experiment")
_STIM_SIM = os.path.join(_ROOT, "stim_simulation", "simulation")
for _p in (_ROOT, _IBM, _STIM_SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_qiskit_stabs(distance: int):
    """Return (x_stabs, z_stabs, logical_z) from the Qiskit generator."""
    from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
    sc = SurfaceCodeCircuit(distance=distance, num_rounds=1)
    return sc.x_stabilizers, sc.z_stabilizers, sc.logical_z


def _build_circuit(distance: int, rounds: int,
                   dp: float, mf: float, rf: float, gd: float) -> stim.Circuit:
    from extractors.stim_compat import build_qiskit_style_stim_circuit
    x_stabs, z_stabs, logical_z = _load_qiskit_stabs(distance)
    return build_qiskit_style_stim_circuit(
        distance=distance,
        num_rounds=rounds,
        x_stabilizers=x_stabs,
        z_stabilizers=z_stabs,
        logical_z_qubits=logical_z,
        noise={"dp": dp, "mf": mf, "rf": rf, "gd": gd},
    )


def create_surface_code_circuit(distance: int, rounds: int, noise: float,
                                 meas_noise: float = 0.0,
                                 reset_noise: float = 0.0,
                                 gate_noise: float = 0.0) -> stim.Circuit:
    """
    Build a Stim circuit in the Qiskit-compatible convention.

    `noise` is interpreted as data-qubit depolarization per round (same role
    as `before_round_data_depolarization` in stim.Circuit.generated).
    """
    return _build_circuit(distance, rounds,
                          dp=noise, mf=meas_noise, rf=reset_noise, gd=gate_noise)


def _num_ancilla(distance: int) -> int:
    x_stabs, z_stabs, _ = _load_qiskit_stabs(distance)
    return len(x_stabs) + len(z_stabs)


def generate_dataset(
    distance: int, rounds: int, noise_rate: float, shots: int,
    error_type: str = "X",
    meas_noise: float = 0.0, reset_noise: float = 0.0, gate_noise: float = 0.0,
    data_depol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (detectors, physical_errors) pairs for ML training.

    Ancilla ordering in the flattened detector vector matches
    SurfaceCodeGraphMapper/SurfaceCodeImageMapper's expectation:
      per round = [X_0..X_{nx-1}, Z_0..Z_{nz-1}]

    Data qubits are indexed 0..d^2-1 (consistent with Qiskit SurfaceCodeCircuit
    and build_qiskit_style_stim_circuit).
    """
    num_data = distance * distance
    num_ancilla = _num_ancilla(distance)
    num_detectors = num_ancilla * rounds

    clean_circuit = _build_circuit(distance, rounds,
                                   dp=data_depol, mf=meas_noise,
                                   rf=reset_noise, gd=gate_noise)
    flat_circuit = clean_circuit.flattened()

    detector_data = np.zeros((shots, num_detectors), dtype=np.float32)
    physical_errors = np.zeros((shots, num_data), dtype=np.float32)
    stim_error_cmd = "X_ERROR" if error_type == "X" else "Z_ERROR"

    data_indices = list(range(num_data))

    for i in range(shots):
        error_mask = np.random.random(num_data) < noise_rate
        physical_errors[i] = error_mask.astype(np.float32)
        error_stim_indices = [data_indices[j] for j in range(num_data) if error_mask[j]]

        noisy_circuit = stim.Circuit()
        injected = False
        for instruction in flat_circuit:
            noisy_circuit.append(instruction)
            if not injected and instruction.name in ["R", "RX", "RY", "RZ",
                                                     "MR", "MRX", "MRY", "MRZ"]:
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
