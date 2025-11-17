import stim
import numpy as np
from typing import Tuple

def create_color_code_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
    """
    Generates a Stim circuit for the 2D Color Code.

    Args:
        distance (int): The code distance (d). Must be an odd integer.
        rounds (int): The number of noisy measurement rounds (T).
        noise (float): The physical error rate (depolarizing noise).

    Returns:
        stim.Circuit: The constructed Stim circuit object.
    """
    return stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise
    )

def generate_dataset(circuit: stim.Circuit, shots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples error syndromes (detectors) and logical errors (observables) from the circuit.

    Args:
        circuit (stim.Circuit): The quantum circuit to simulate.
        shots (int): The number of experimental shots (samples) to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - detector_data (np.ndarray): Boolean array of shape [shots, num_detectors].
            - observable_data (np.ndarray): Boolean array of shape [shots, num_observables].
    """
    sampler = circuit.compile_detector_sampler()
    
    # 'bit_packed=False' ensures the output is a byte array (0s and 1s), 
    # which is easier to handle for ML preprocessing than packed bits.
    detector_data, observable_data = sampler.sample(
        shots=shots,
        bit_packed=False,
        separate_observables=True 
    )
    
    return detector_data, observable_data