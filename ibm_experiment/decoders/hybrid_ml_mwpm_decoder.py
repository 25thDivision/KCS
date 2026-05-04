"""
Hybrid ML → MWPM Residual Decoder (역순 하이브리드).

기존 HybridMWPMDecoder가 (MWPM → ML residual)이라면, 본 클래스는
(ML → MWPM residual) 순서로 동작한다.

2-Stage:
  Stage 1: ML correction (full syndrome 기반)
  Stage 2: corrected_data = data ^ ml_correction
           residual_z = H_z @ corrected_data % 2
           residual=0인 shot은 early exit
  Stage 3: residual_z를 단일-라운드 Z-syndrome으로 보고 MWPM 디코딩
           heavyhex: MWPMHeavyHexDecoder.decode_z_syndrome (lookup)
           surface_code: H_z 기반 pymatching.Matching
           최종 = ml ^ mwpm_residual

Notes:
  ML decoder를 다시 호출하지 않도록 파이프라인에서 이미 계산된
  ml_corrections를 직접 받는다.
"""

import numpy as np


class HybridMLMWPMDecoder:
    """
    Generic ML → Residual Z-Syndrome → MWPM hybrid decoder.

    Parameters:
        distance: code distance
        converter: StimFormatConverter
        code_type: "heavyhex_surface_code" or "surface_code"
                   (None이면 converter.code_type을 사용)
    """

    def __init__(self, distance: int, converter, code_type: str = None):
        self.distance = distance
        self.converter = converter
        self.code_type = code_type or getattr(converter, "code_type", None)

        if self.code_type == "heavyhex_surface_code":
            from decoders.mwpm_heavyhex_decoder import MWPMHeavyHexDecoder
            self.residual_mwpm = MWPMHeavyHexDecoder(distance=distance)
            self.h_z = self.residual_mwpm.h_z          # (4, 9)
            self.num_data = self.residual_mwpm.num_data
            self.residual_matcher = None
        elif self.code_type == "surface_code":
            import pymatching
            from scipy.sparse import csc_matrix
            z_stabs = converter._z_stabs
            self.num_data = converter.num_data_qubits
            h_z = np.zeros((len(z_stabs), self.num_data), dtype=np.uint8)
            for i, stab in enumerate(z_stabs):
                for q in stab:
                    h_z[i, q] = 1
            self.h_z = h_z
            self.residual_mwpm = None
            self.residual_matcher = pymatching.Matching(csc_matrix(h_z))
        else:
            raise ValueError(f"Unsupported code_type: {self.code_type!r}")

        self.num_z_stabs = self.h_z.shape[0]

    def decode(self, data_states: np.ndarray,
               ml_corrections: np.ndarray) -> np.ndarray:
        """
        Args:
            data_states: (N, num_data)
            ml_corrections: (N, num_data) — 파이프라인이 미리 계산한 ML 출력

        Returns:
            corrections: (N, num_data) int8
        """
        ml = np.asarray(ml_corrections, dtype=np.uint8)
        if ml.shape[1] > self.num_data:
            ml = ml[:, : self.num_data]
        N = ml.shape[0]
        if data_states.shape != ml.shape:
            raise ValueError(
                f"data_states shape {data_states.shape} != ml shape {ml.shape}"
            )

        # Stage 2: Residual Z-syndrome
        corrected_data = data_states.astype(np.uint8) ^ ml
        residual_z = (self.h_z @ corrected_data.T % 2).T  # (N, num_z_stabs)

        needs_mwpm = residual_z.any(axis=1)
        nm = int(needs_mwpm.sum())

        print(f"        [Hybrid ML+MWPM] ML solved: {N - nm}/{N}, "
              f"MWPM needed: {nm}/{N}")

        if nm == 0:
            return ml.astype(np.int8)

        idxs = np.where(needs_mwpm)[0]
        residual = residual_z[idxs].astype(np.uint8)

        # Stage 3: MWPM on residual Z-syndrome
        if self.code_type == "heavyhex_surface_code":
            mwpm_resid = self.residual_mwpm.decode_z_syndrome(residual)
        else:
            mwpm_resid = np.zeros((len(idxs), self.num_data), dtype=np.int8)
            for k, syn in enumerate(residual):
                mwpm_resid[k] = self.residual_matcher.decode(syn).astype(np.int8)

        combined = ml.copy()
        combined[idxs] = (ml[idxs] ^ mwpm_resid.astype(np.uint8))
        return combined.astype(np.int8)
