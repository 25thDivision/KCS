"""
Hybrid ML → MWPM Residual Decoder for color code (역순 하이브리드).

기존 HybridMWPMDecoder가 (MWPM → ML residual)이라면, 본 클래스는
(ML → MWPM residual) 순서로 동작한다.

2-Stage:
  Stage 1: ML correction (full syndrome 기반)
  Stage 2: corrected_data = data ^ ml_correction
           residual_z = H @ corrected_data % 2  (color code Z-syndrome)
           residual=0인 shot은 early exit
  Stage 3: residual_z를 단일 Z-syndrome으로 보고 MWPM Restriction 디코딩
           (MWPMColorCodeDecoder._decode_single 재사용)
           최종 = ml ^ mwpm_residual

Notes:
  ML decoder를 다시 호출하지 않도록 파이프라인에서 이미 계산된
  ml_corrections를 직접 받는다.
"""

import numpy as np


class HybridMLMWPMDecoder:
    """ML → Residual Z-Syndrome → MWPM Restriction 하이브리드."""

    def __init__(self, distance: int):
        from decoders.mwpm_colorcode_decoder import MWPMColorCodeDecoder
        self.distance = distance
        self.residual_mwpm = MWPMColorCodeDecoder(distance=distance)
        self.h_matrix = self.residual_mwpm.h_matrix     # (num_faces, num_data)
        self.num_data = self.residual_mwpm.num_data
        self.num_faces = self.residual_mwpm.num_faces

    def decode(self, data_states: np.ndarray,
               ml_corrections: np.ndarray) -> np.ndarray:
        """
        Args:
            data_states: (N, num_data)
            ml_corrections: (N, num_data)

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
        residual_z = (self.h_matrix @ corrected_data.T % 2).T  # (N, num_faces)

        needs_mwpm = residual_z.any(axis=1)
        nm = int(needs_mwpm.sum())

        print(f"        [Hybrid ML+MWPM] ML solved: {N - nm}/{N}, "
              f"MWPM needed: {nm}/{N}")

        if nm == 0:
            return ml.astype(np.int8)

        idxs = np.where(needs_mwpm)[0]
        mwpm_resid = np.zeros((len(idxs), self.num_data), dtype=np.int8)
        for k, gi in enumerate(idxs):
            mwpm_resid[k] = self.residual_mwpm._decode_single(
                residual_z[gi].astype(np.uint8)
            )

        combined = ml.copy()
        combined[idxs] = (ml[idxs] ^ mwpm_resid.astype(np.uint8))
        return combined.astype(np.int8)
