"""
Hybrid MWPM + ML Residual Decoder for IBM Heavy-Hex Surface Code.

2-Stage:
  Stage 1: MWPM이 cumulative Z-syndrome 기반으로 correction 생성
  Stage 2: corrected_data = data ^ mwpm_correction
           residual_z = H_z @ corrected_data % 2
           residual=0인 shot은 early exit
  Stage 3: residual을 ML 입력 형식으로 구성 → ML inference
           최종 = mwpm ^ ml_residual
"""

import numpy as np

from decoders.mwpm_heavyhex_decoder import MWPMHeavyHexDecoder


class HybridMWPMDecoder:
    """
    IBM Heavy-Hex 2-Stage Decoder: MWPM → Residual Z-Syndrome → ML.

    Parameters:
        distance: 3
        ml_decoder: MLDecoderAdapter
        converter: StimFormatConverter (graph/image mapper 접근용)
        model_type: "graph" or "image"
    """

    def __init__(self, distance: int, ml_decoder, converter, model_type: str):
        self.distance = distance
        self.ml_decoder = ml_decoder
        self.converter = converter
        self.model_type = model_type

        self.mwpm = MWPMHeavyHexDecoder(distance=distance)
        self.h_z = self.mwpm.h_z              # (4, 9)
        self.num_z_stabs = self.h_z.shape[0]  # 4
        self.num_data = self.mwpm.num_data    # 9
        self.num_rounds = converter.num_rounds
        self.num_stabilizers = converter.num_stabilizers  # 8

    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        Args:
            syndromes: (N, num_rounds, 8) HW raw order
            data_states: (N, 9)
        Returns:
            corrections: (N, 9)
        """
        N = syndromes.shape[0]

        # Stage 1: MWPM
        correction_mwpm = self.mwpm.decode(syndromes, data_states)  # (N, 9)

        # Stage 2: Residual Z-syndrome (Stim order, cumulative)
        corrected_data = (data_states.astype(np.uint8) ^ correction_mwpm.astype(np.uint8))
        residual_z = (self.h_z @ corrected_data.T % 2).T  # (N, 4)

        needs_ml = residual_z.any(axis=1)
        ml_count = int(needs_ml.sum())

        print(f"        [Hybrid] MWPM solved: {N - ml_count}/{N}, ML needed: {ml_count}/{N}")

        if ml_count == 0:
            return correction_mwpm

        ml_indices = np.where(needs_ml)[0]
        n_ml = ml_count

        # Stage 2.5: ML 입력 구성 (HW→Stim 변환을 우회하고 직접 stim detector 구성)
        # detector layout: (n_ml, num_rounds * 8), Stim 순서, Z bits = first 4
        stim_detectors = np.zeros(
            (n_ml, self.num_rounds * self.num_stabilizers), dtype=np.float32
        )
        # 마지막 라운드의 Z-stab 부분 (Stim의 0-3)에 residual 주입
        last_offset = (self.num_rounds - 1) * self.num_stabilizers
        stim_detectors[:, last_offset : last_offset + self.num_z_stabs] = (
            residual_z[ml_indices].astype(np.float32)
        )

        # Stage 3: ML inference (mapper 직접 호출)
        if self.model_type == "graph":
            node_features = self.converter.graph_mapper.map_to_node_features(stim_detectors)
            correction_ml = self.ml_decoder.decode(node_features, edge_index=self.converter.edge_index)
        else:
            images = self.converter.image_mapper.map_to_images(stim_detectors)
            correction_ml = self.ml_decoder.decode(images)

        if correction_ml.shape[1] > self.num_data:
            correction_ml = correction_ml[:, : self.num_data]

        # 합성
        combined = correction_mwpm.copy()
        combined[ml_indices] = (
            correction_mwpm[ml_indices].astype(np.uint8) ^ correction_ml.astype(np.uint8)
        ).astype(np.int8)

        return combined
