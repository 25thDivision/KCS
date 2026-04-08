"""
Color Code MWPM Restriction Decoder (d=3, d=5 범용)

Restriction Decoder (Delfosse, 2014):
  1. Face 정보에서 H matrix (num_faces x num_data) 동적 구축
  2. Face color에서 3개 restriction (RG, RB, GB) sub-H matrix 구성
  3. 각 restriction에서 PyMatching MWPM 실행
  4. Full syndrome consistency check -> minimum weight 선택

d=3 [[7,1,3]] Steane Code:
  3 faces (R, G, B), 7 data qubits
  H 열이 모두 distinct -> single error lookup fallback 가능

d=5 [[19,1,5]] Color Code:
  9 faces (R, R, G, G, B, B, R, G, B), 19 data qubits
  Inconsistent 시 minimum weight 선택
"""

import numpy as np
from scipy.sparse import csc_matrix

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False
    print("[WARNING] pymatching not installed. MWPMColorCodeDecoder will use fallback (d=3 only).")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from circuits.qiskit_colorcode_generator import STEANE_CODE, D5_COLOR_CODE


# 3개의 색상 쌍
COLOR_PAIRS = [("RG", "R", "G"), ("RB", "R", "B"), ("GB", "G", "B")]


class MWPMColorCodeDecoder:
    """
    Color Code MWPM Restriction Decoder (d=3, d=5 범용).

    생성자에서:
      - Face 정보 로드 (STEANE_CODE or D5_COLOR_CODE)
      - H matrix 동적 구축
      - 3개 restriction sub-H + PyMatching Matching 생성
      - d=3: single error lookup table 캐시

    decode(syndromes, data_states):
      - Z-syndrome 추출 (마지막 라운드, Z-type 부분)
      - Restriction MWPM + consistency check
      - Return corrections (N, num_data)
    """

    def __init__(self, distance: int):
        self.distance = distance

        # Face 정보 로드
        if distance == 3:
            code_def = STEANE_CODE
            faces = [s for s in code_def["stabilizers"] if s["type"] == "X"]
        elif distance == 5:
            code_def = D5_COLOR_CODE
            faces = code_def["plaquettes"]
        else:
            raise ValueError(f"Distance {distance} not supported. Use 3 or 5.")

        self.num_data = code_def["num_data"]
        self.num_faces = len(faces)
        self.faces = faces

        # H matrix 구축 (num_faces x num_data)
        self.h_matrix = np.zeros((self.num_faces, self.num_data), dtype=np.uint8)
        self.face_colors = []
        for i, face in enumerate(faces):
            for q in face["qubits"]:
                self.h_matrix[i, q] = 1
            self.face_colors.append(face["color"])

        # 3개 restriction sub-H matrix + PyMatching matcher
        self.restrictions = {}
        self.matchers = {}

        for pair_name, c1, c2 in COLOR_PAIRS:
            row_indices = [i for i, c in enumerate(self.face_colors) if c in (c1, c2)]
            sub_h = self.h_matrix[row_indices, :]
            self.restrictions[pair_name] = {
                "row_indices": row_indices,
                "sub_h": sub_h,
            }

            if HAS_PYMATCHING:
                self.matchers[pair_name] = pymatching.Matching(csc_matrix(sub_h))

        # single error lookup table (fallback용)
        self._single_error_table = {}
        for q in range(self.num_data):
            syn_tuple = tuple(self.h_matrix[:, q].tolist())
            self._single_error_table[syn_tuple] = q

    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        MWPM Restriction Decoder로 correction을 계산합니다.

        Args:
            syndromes: (N, num_rounds, num_stabilizers_per_round)
                       num_stabilizers_per_round = num_faces * 2 (X first, Z second)
            data_states: (N, num_data)

        Returns:
            corrections: (N, num_data)
        """
        # Z-syndrome 추출: 마지막 라운드, Z-type 부분 (뒤쪽 num_faces개)
        if syndromes.ndim == 3:
            z_syn = syndromes[:, -1, self.num_faces:].astype(np.uint8)
        elif syndromes.ndim == 2:
            z_syn = syndromes[:, self.num_faces:].astype(np.uint8)
        else:
            raise ValueError(f"Unexpected syndromes shape: {syndromes.shape}")

        N = z_syn.shape[0]
        corrections = np.zeros((N, self.num_data), dtype=np.int8)

        for i in range(N):
            corrections[i] = self._decode_single(z_syn[i])

        return corrections

    def _decode_single(self, z_syndrome: np.ndarray) -> np.ndarray:
        """
        단일 shot에 대한 restriction MWPM 디코딩.

        1. 3개 restriction 각각에서 MWPM 실행
        2. Full consistency check: (H @ correction) % 2 == z_syndrome
        3. Consistent corrections 중 minimum weight 선택
        4. 모두 inconsistent: d=3 -> direct lookup, d=5 -> minimum weight
        """
        if z_syndrome.sum() == 0:
            return np.zeros(self.num_data, dtype=np.int8)

        if not HAS_PYMATCHING:
            if self._single_error_table is not None:
                return self._direct_lookup(z_syndrome)
            return np.zeros(self.num_data, dtype=np.int8)

        # 3개 restriction에서 MWPM 실행
        consistent = []
        inconsistent = []

        for pair_name, info in self.restrictions.items():
            row_indices = info["row_indices"]
            restricted_syn = z_syndrome[row_indices].astype(np.uint8)

            if restricted_syn.sum() == 0:
                continue

            correction = self.matchers[pair_name].decode(restricted_syn)

            # Full consistency check: H @ correction == z_syndrome (mod 2)
            predicted_syn = (self.h_matrix @ correction) % 2
            weight = int(correction.sum())

            if np.array_equal(predicted_syn, z_syndrome):
                consistent.append((weight, correction.astype(np.int8)))
            else:
                inconsistent.append((weight, correction.astype(np.int8)))

        # Consistent corrections 중 minimum weight 선택
        if consistent:
            consistent.sort(key=lambda x: x[0])
            return consistent[0][1]

        # Fallback: d=3 -> direct lookup, d=5 -> minimum weight
        if self._single_error_table is not None:
            return self._direct_lookup(z_syndrome)

        # d=5: inconsistent 중 minimum weight
        if inconsistent:
            inconsistent.sort(key=lambda x: x[0])
            return inconsistent[0][1]

        return np.zeros(self.num_data, dtype=np.int8)

    def _direct_lookup(self, z_syndrome: np.ndarray) -> np.ndarray:
        """d=3 single error direct lookup fallback."""
        correction = np.zeros(self.num_data, dtype=np.int8)
        syn_tuple = tuple(z_syndrome.astype(int).tolist())
        error_qubit = self._single_error_table.get(syn_tuple)
        if error_qubit is not None:
            correction[error_qubit] = 1
        return correction


# ==============================================================================
# Verification
# ==============================================================================
def verify_decoder():
    """d=3, d=5 디코더 검증."""
    print("=" * 60)
    print("  MWPM Restriction Decoder Verification")
    print("=" * 60)
    print(f"  PyMatching available: {HAS_PYMATCHING}")

    all_pass = True

    for d in [3, 5]:
        decoder = MWPMColorCodeDecoder(distance=d)
        print(f"\n  --- d={d}: {decoder.num_faces} faces, {decoder.num_data} data qubits ---")
        print(f"  H matrix shape: {decoder.h_matrix.shape}")
        print(f"  Face colors: {decoder.face_colors}")

        for pair_name, info in decoder.restrictions.items():
            sub_h = info["sub_h"]
            max_col_weight = sub_h.sum(axis=0).max()
            print(f"  Restriction {pair_name}: {sub_h.shape}, max col weight={max_col_weight}")

        # Single qubit error test
        print(f"\n  [Test] Single-qubit Z-error detection (Z-syndrome):")
        for q in range(decoder.num_data):
            # Z-syndrome for X-error on qubit q
            z_syn = decoder.h_matrix[:, q].astype(np.uint8)
            # Build full syndrome: X-part=0, Z-part=z_syn
            full_syn = np.zeros(decoder.num_faces * 2, dtype=np.uint8)
            full_syn[decoder.num_faces:] = z_syn

            syndromes = full_syn.reshape(1, 1, -1)
            data_states = np.zeros((1, decoder.num_data), dtype=np.int8)
            corrections = decoder.decode(syndromes, data_states)

            corrected_q = np.where(corrections[0] == 1)[0]
            # Check: correction must produce same syndrome as original error
            if corrections[0].sum() > 0:
                corr_syn = (decoder.h_matrix @ corrections[0].astype(np.uint8)) % 2
                syn_match = np.array_equal(corr_syn, z_syn)
            else:
                syn_match = (z_syn.sum() == 0)

            status = "OK" if syn_match else "FAIL"
            if not syn_match:
                all_pass = False
            print(f"    [{status}] q{q}: syn={list(z_syn)} -> correction={list(corrected_q)} "
                  f"(weight={corrections[0].sum()})")

    # Zero syndrome test
    print(f"\n  [Test] Zero syndrome:")
    for d in [3, 5]:
        decoder = MWPMColorCodeDecoder(distance=d)
        nf = decoder.num_faces
        syn_zero = np.zeros((1, 1, nf * 2), dtype=np.uint8)
        data_zero = np.zeros((1, decoder.num_data), dtype=np.int8)
        corr = decoder.decode(syn_zero, data_zero)
        ok = np.all(corr == 0)
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{status}] d={d}: zero syndrome -> zero correction")

    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    verify_decoder()
