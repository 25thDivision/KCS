"""
MWPM baseline decoder for rotated surface code.

Builds a Stim reference circuit that matches the Qiskit SurfaceCodeCircuit's
stabilizer schedule (X-stabs then Z-stabs per round, logical Z on the left
column), extracts its decomposed detector error model, and wires it into a
`pymatching.Matching` graph. Decoding predicts the logical-Z observable; the
returned correction flips one data qubit in the logical-Z support to realise
that flip, keeping the interface identical to the ML decoders.

Expected syndrome format:
  detection_events of shape (N, num_detectors) where num_detectors matches the
  DEM produced by the reference circuit. Use
  `StimFormatConverter.hw_to_mwpm_detectors()` in stim_compat.py to convert
  Qiskit (syndromes, data_states) to this format — the two share the same
  reference-circuit builder so their detector orderings are consistent.
"""

import os
import re
import sys
from typing import Union

import numpy as np

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False

_this_dir = os.path.dirname(os.path.abspath(__file__))
_ibm_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_ibm_dir)
_stim_sim = os.path.join(_root_dir, "stim_simulation", "simulation")
if _stim_sim not in sys.path:
    sys.path.append(_stim_sim)
if _ibm_dir not in sys.path:
    sys.path.append(_ibm_dir)


def _parse_noise_profile(profile: Union[str, dict]) -> dict:
    """
    Accepts either a dict {"dp":..., "mf":..., "rf":..., "gd":...} or a string
    shaped like "realistic/dp0.001_mf0.01_rf0.01_gd0.008" (the subdir name
    used elsewhere in the repo).
    """
    if isinstance(profile, dict):
        return {
            "dp": float(profile.get("dp", 0.0)),
            "mf": float(profile.get("mf", 0.0)),
            "rf": float(profile.get("rf", 0.0)),
            "gd": float(profile.get("gd", 0.0)),
        }
    if isinstance(profile, str):
        tail = profile.split("/")[-1]
        out = {"dp": 0.0, "mf": 0.0, "rf": 0.0, "gd": 0.0}
        for key in out:
            m = re.search(rf"{key}([0-9]+(?:\.[0-9]+)?)", tail)
            if m:
                out[key] = float(m.group(1))
        return out
    raise TypeError(f"noise_profile must be dict or str, got {type(profile)}")


class MWPMDecoder:
    """
    PyMatching-based MWPM decoder for rotated surface code memory-Z experiments.
    """

    def __init__(self, distance: int, rounds: int, noise_profile: Union[str, dict]):
        if not HAS_STIM:
            raise ImportError("stim is required for MWPMDecoder.")
        if not HAS_PYMATCHING:
            raise ImportError("pymatching is required for MWPMDecoder.")

        self.distance = distance
        self.rounds = rounds
        self.num_data = distance ** 2
        self.noise = _parse_noise_profile(noise_profile)

        # Build a Stim reference circuit with the SAME stabilizer schedule as
        # the Qiskit SurfaceCodeCircuit so that the DEM's detector order and
        # observable definition match StimFormatConverter's m2d converter.
        from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
        from extractors.stim_compat import build_qiskit_style_stim_circuit
        sc = SurfaceCodeCircuit(distance=distance, num_rounds=rounds)
        self.circuit = build_qiskit_style_stim_circuit(
            distance=distance,
            num_rounds=rounds,
            x_stabilizers=sc.x_stabilizers,
            z_stabilizers=sc.z_stabilizers,
            logical_z_qubits=sc.logical_z,
            noise=self.noise,
        )

        dem = self.circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.matcher = pymatching.Matching.from_detector_error_model(dem)

        # Logical Z support in Qiskit data-qubit indexing (left column of dxd
        # data grid). Flipping any single qubit in this support implements a
        # logical-Z flip.
        self.logical_z_qubits = list(range(0, distance ** 2, distance))

        print(f"[MWPMDecoder] d={distance}, rounds={rounds}")
        print(f"    noise: dp={self.noise['dp']}, mf={self.noise['mf']}, "
              f"rf={self.noise['rf']}, gd={self.noise['gd']}")
        print(f"    detectors={self.num_detectors}, "
              f"observables={self.num_observables}")

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """
        Decode detector syndromes in Stim format.

        Args:
            detection_events: (N, num_detectors) binary array matching the
                              decomposed DEM of the generated stim circuit.

        Returns:
            corrections: (N, num_data) int8 array. Encodes the MWPM-predicted
                         logical-Z flip by flipping a single data qubit in the
                         logical-Z support; the LogicalErrorRateEvaluator then
                         XORs this against the measured data bits.
        """
        det = np.asarray(detection_events, dtype=np.uint8)
        if det.shape[1] != self.num_detectors:
            raise ValueError(
                f"detection_events has {det.shape[1]} columns but DEM expects "
                f"{self.num_detectors}."
            )

        predictions = self.matcher.decode_batch(det)  # (N, num_observables)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        logical_flip = predictions[:, 0].astype(np.int8)

        N = det.shape[0]
        corrections = np.zeros((N, self.num_data), dtype=np.int8)
        corrections[:, self.logical_z_qubits[0]] = logical_flip
        return corrections


if __name__ == "__main__":
    print("=" * 60)
    print("  MWPMDecoder self-test")
    print("=" * 60)
    dec = MWPMDecoder(distance=3, rounds=3,
                      noise_profile={"dp": 0.01, "mf": 0.005,
                                     "rf": 0.005, "gd": 0.004})
    zero = np.zeros((4, dec.num_detectors), dtype=np.uint8)
    corr = dec.decode_batch(zero)
    print(f"    zero syndrome -> correction sum: {corr.sum()} (expected 0)")
