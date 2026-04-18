"""
Heavy-Hex Surface Code d=3 MWPM Decoder.

PyMatching 기반 단순 H-matrix 디코더 (color code restriction decoder와 다름).
Z-stabilizer (4개) 만으로 X-error 교정.

H_z는 generators/heavyhex_surface_code.HEAVYHEX_D3에서 동적으로 구축.
Stim 순서 [Z{0134}, Z{1245}, Z{01}, Z{78}].
"""

import os
import sys
from itertools import product

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
ibm_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(ibm_dir)
stim_sim_dir = os.path.join(root_dir, "stim_simulation", "simulation")
sys.path.append(stim_sim_dir)

from generators.heavyhex_surface_code import HEAVYHEX_D3, reorder_hw_to_stim


class MWPMHeavyHexDecoder:
    """Heavy-hex surface code d=3 MWPM Z-error decoder."""

    def __init__(self, distance: int = 3):
        if distance != 3:
            raise NotImplementedError(f"Only d=3 supported, got d={distance}")

        self.distance = distance
        cfg = HEAVYHEX_D3
        self.num_data = cfg["num_data"]
        z_stabs_dict = cfg["z_stabilizers"]
        z_ancilla = cfg["z_ancilla"]

        # H_z in Stim order: rows = Z-ancilla in z_ancilla list order
        self.h_z = np.zeros((len(z_ancilla), self.num_data), dtype=np.uint8)
        for i, anc in enumerate(z_ancilla):
            for q in z_stabs_dict[anc]:
                self.h_z[i, q] = 1

        # PyMatching이 column weight>2를 거부하므로 minimum-weight lookup table 구축.
        # 4 Z-stabs → 16 syndromes, 9 data → 512 patterns. 한 번만 enumerate.
        num_syn = 1 << len(z_ancilla)
        self._lookup = np.zeros((num_syn, self.num_data), dtype=np.int8)
        best_w = np.full(num_syn, self.num_data + 1, dtype=np.int32)
        best_w[0] = 0
        for bits in product([0, 1], repeat=self.num_data):
            corr = np.array(bits, dtype=np.uint8)
            syn = (self.h_z @ corr) % 2
            idx = int(np.packbits(syn, bitorder="little")[0]) if len(syn) <= 8 else int("".join(map(str, syn[::-1])), 2)
            w = int(corr.sum())
            if w < best_w[idx]:
                best_w[idx] = w
                self._lookup[idx] = corr.astype(np.int8)

    def _to_cumulative_z_stim(self, syndromes: np.ndarray) -> np.ndarray:
        """
        HW raw syndromes (N, num_rounds, 8) → cumulative Z-syndrome (N, 4) in Stim order.

        no-reset XOR tracking → temporal differencing → XOR-sum across rounds
        = total accumulated Z-events. Equivalent to raw last-round HW + reorder.
        """
        N, num_rounds, _ = syndromes.shape

        # Temporal differencing
        differenced = np.zeros_like(syndromes)
        differenced[:, 0, :] = syndromes[:, 0, :]
        if num_rounds > 1:
            differenced[:, 1:, :] = (
                syndromes[:, 1:, :] != syndromes[:, :-1, :]
            ).astype(syndromes.dtype)

        # HW → Stim 순서 재배열
        flat = differenced.reshape(N, -1)
        stim_flat = reorder_hw_to_stim(flat, num_rounds).reshape(N, num_rounds, 8)

        # 누적: XOR-sum across rounds, take Z-bits (first 4 in Stim order)
        cumulative = stim_flat.sum(axis=1) % 2
        return cumulative[:, :4].astype(np.uint8)

    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        Args:
            syndromes: (N, num_rounds, 8) HW order raw cumulative
            data_states: (N, 9)
        Returns:
            corrections: (N, 9) int8
        """
        z_syn = self._to_cumulative_z_stim(syndromes)
        return self.decode_z_syndrome(z_syn)

    def decode_z_syndrome(self, z_syn: np.ndarray) -> np.ndarray:
        """
        Direct Z-syndrome lookup (Hybrid의 residual path용).

        Args:
            z_syn: (N, 4) Stim 순서 Z-syndrome
        Returns:
            corrections: (N, 9)
        """
        z_syn = z_syn.astype(np.uint8)
        # syndrome bits → integer index (LSB = bit 0)
        weights = np.array([1, 2, 4, 8], dtype=np.int32)
        indices = (z_syn @ weights).astype(np.int32)
        return self._lookup[indices].copy()


if __name__ == "__main__":
    print("=" * 60)
    print("  MWPMHeavyHexDecoder Verification")
    print("=" * 60)
    dec = MWPMHeavyHexDecoder(distance=3)
    print(f"  H_z shape: {dec.h_z.shape}")
    print(f"  H_z =\n{dec.h_z}")

    # Zero syndrome
    syn = np.zeros((1, 3, 8), dtype=np.float32)
    data = np.zeros((1, 9), dtype=np.int8)
    corr = dec.decode(syn, data)
    print(f"  [Zero] correction sum: {corr.sum()} (expected 0)")

    # Single qubit X-error: build raw syndrome that decodes to that qubit
    print(f"\n  Single-qubit error tests (direct Z-syndrome):")
    for q in range(9):
        z_syn = dec.h_z[:, q].reshape(1, -1).astype(np.uint8)
        corr = dec.decode_z_syndrome(z_syn)
        recovered = (dec.h_z @ corr[0].astype(np.uint8)) % 2
        ok = np.array_equal(recovered, z_syn[0])
        print(f"    q{q}: syn={list(z_syn[0])} → corr={list(np.where(corr[0])[0])} [{'OK' if ok else 'FAIL'}]")
