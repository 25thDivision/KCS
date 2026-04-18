"""
Hybrid MWPM + ML Residual Decoder

2-Stage Decoder:
  Stage 1: MWPM이 Z-syndrome 기반으로 correction 생성
  Stage 2: MWPM correction을 data에 적용 후 residual syndrome 계산
           → residual이 0이면 early exit (MWPM만 사용)
           → residual이 non-zero인 shot만 ML에 넘김
  Stage 3: MWPM correction ^ ML correction = 최종 correction
"""

import numpy as np
from decoders.mwpm_colorcode_decoder import MWPMColorCodeDecoder


class HybridMWPMDecoder:
    """
    2-Stage Decoder: MWPM → Residual Syndrome → ML

    Parameters:
        distance: Code distance (3 or 5)
        ml_decoder: MLDecoderAdapter 인스턴스
        converter: StimFormatConverter 인스턴스
        model_type: "graph" or "image"
    """

    def __init__(self, distance: int, ml_decoder, converter, model_type: str):
        self.distance = distance
        self.ml_decoder = ml_decoder
        self.converter = converter
        self.model_type = model_type

        self.mwpm = MWPMColorCodeDecoder(distance=distance)
        self.h_matrix = self.mwpm.h_matrix       # (num_faces, num_data)
        self.num_faces = self.mwpm.num_faces
        self.num_data = self.mwpm.num_data

    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        Hybrid MWPM+ML 디코딩.

        Args:
            syndromes: (N, num_rounds, num_stabilizers) or (N, num_stabilizers)
            data_states: (N, num_data)

        Returns:
            corrections: (N, num_data)
        """
        N = syndromes.shape[0]

        # Stage 1: MWPM decode
        correction_mwpm = self.mwpm.decode(syndromes, data_states)  # (N, num_data)

        # Stage 2: Residual Z-syndrome 계산
        corrected_data = (data_states ^ correction_mwpm).astype(np.uint8)
        residual_z_syn = (self.h_matrix @ corrected_data.T % 2).T  # (N, num_faces)

        # Early exit 분기
        needs_ml = residual_z_syn.any(axis=1)  # (N,) bool
        ml_count = int(needs_ml.sum())

        print(f"    [Hybrid] MWPM solved: {N - ml_count}/{N} shots, ML needed: {ml_count}/{N}")

        if ml_count == 0:
            return correction_mwpm

        ml_indices = np.where(needs_ml)[0]

        # Stage 2.5: ML용 residual syndrome 구성 (needs_ml인 shot만)
        residual_syndromes = syndromes[ml_indices].copy()

        if syndromes.ndim == 3:
            # 마지막 라운드의 Z-syndrome 부분을 residual로 교체
            residual_syndromes[:, -1, self.num_faces:] = residual_z_syn[ml_indices]
        elif syndromes.ndim == 2:
            residual_syndromes[:, self.num_faces:] = residual_z_syn[ml_indices]

        # Stage 3: ML inference (ml_indices만)
        if self.model_type == "graph":
            model_input, edge_index = self.converter.to_graph_format(residual_syndromes)
            correction_ml = self.ml_decoder.decode(model_input, edge_index=edge_index)
        else:
            model_input = self.converter.to_image_format(residual_syndromes)
            correction_ml = self.ml_decoder.decode(model_input)

        # Truncate: ML 출력이 num_data보다 클 수 있음
        if correction_ml.shape[1] > self.num_data:
            correction_ml = correction_ml[:, :self.num_data]

        # 합성: MWPM ^ ML
        combined = correction_mwpm.copy()
        combined[ml_indices] = (correction_mwpm[ml_indices] ^ correction_ml).astype(np.int8)

        return combined
