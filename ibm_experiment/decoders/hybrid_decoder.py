"""
Hybrid MWPM + ML Residual Decoder.

2-Stage:
  Stage 1: MWPM correction
    - heavyhex_surface_code: 내장 MWPMHeavyHexDecoder를 직접 호출
    - surface_code: per-noise MWPMDecoder가 미리 계산한 corrections를 주입
  Stage 2: corrected_data = data ^ mwpm_correction
           residual_z = H_z @ corrected_data % 2
           residual=0인 shot은 early exit
  Stage 3: residual을 ML 입력 형식으로 구성 → ML inference
           최종 = mwpm ^ ml_residual

Per-round stabilizer layout (residual 주입 위치):
  - heavyhex: reorder_hw_to_stim 결과로 [Z..., X...] 순서 → 오프셋 0
  - surface_code: SurfaceCodeGraphMapper가 [X..., Z...] 순서 → 오프셋 n_x
"""

import numpy as np


class HybridMWPMDecoder:
    """
    Generic MWPM → Residual Z-Syndrome → ML hybrid decoder.

    Parameters:
        distance: code distance
        ml_decoder: MLDecoderAdapter
        converter: StimFormatConverter (mapper / stabilizer 정보 접근용)
        model_type: "graph" or "image"
        code_type: "heavyhex_surface_code" or "surface_code"
                   (None이면 converter.code_type을 사용)
    """

    def __init__(self, distance: int, ml_decoder, converter, model_type: str,
                 code_type: str = None):
        self.distance = distance
        self.ml_decoder = ml_decoder
        self.converter = converter
        self.model_type = model_type
        self.code_type = code_type or getattr(converter, "code_type", None)

        self.num_rounds = converter.num_rounds
        self.num_stabilizers = converter.num_stabilizers

        if self.code_type == "heavyhex_surface_code":
            from decoders.mwpm_heavyhex_decoder import MWPMHeavyHexDecoder
            self.mwpm = MWPMHeavyHexDecoder(distance=distance)
            self.h_z = self.mwpm.h_z              # (4, 9)
            self.num_data = self.mwpm.num_data    # 9
            # heavyhex ML 입력은 reorder_hw_to_stim으로 Z-stabs가 라운드 앞에 배치됨
            self.residual_offset_in_round = 0
        elif self.code_type == "surface_code":
            self.mwpm = None  # caller가 mwpm_corrections를 직접 전달
            z_stabs = converter._z_stabs
            x_stabs = converter._x_stabs
            self.num_data = converter.num_data_qubits
            h_z = np.zeros((len(z_stabs), self.num_data), dtype=np.uint8)
            for i, stab in enumerate(z_stabs):
                for q in stab:
                    h_z[i, q] = 1
            self.h_z = h_z
            # surface_code ML 입력은 SurfaceCodeGraphMapper의 [X..., Z...] 순서
            self.residual_offset_in_round = len(x_stabs)
        else:
            raise ValueError(f"Unsupported code_type: {self.code_type!r}")

        self.num_z_stabs = self.h_z.shape[0]

    def decode(self, syndromes: np.ndarray, data_states: np.ndarray,
               mwpm_corrections: np.ndarray = None) -> np.ndarray:
        """
        Args:
            syndromes: (N, num_rounds, num_stab) HW raw order
            data_states: (N, num_data)
            mwpm_corrections: optional (N, num_data). surface_code에서는 필수,
                              heavyhex_surface_code에서는 None이면 내장 MWPM이
                              호출됨.

        Returns:
            corrections: (N, num_data)
        """
        N = syndromes.shape[0]

        # Stage 1: MWPM correction 확보
        if mwpm_corrections is None:
            if self.mwpm is None:
                raise ValueError(
                    f"mwpm_corrections must be supplied for code_type="
                    f"{self.code_type!r}."
                )
            correction_mwpm = self.mwpm.decode(syndromes, data_states)
        else:
            correction_mwpm = np.asarray(mwpm_corrections, dtype=np.int8)
            if correction_mwpm.shape != (N, self.num_data):
                raise ValueError(
                    f"mwpm_corrections shape {correction_mwpm.shape} != "
                    f"expected ({N}, {self.num_data})."
                )

        # Stage 2: Residual Z-syndrome
        corrected_data = (
            data_states.astype(np.uint8) ^ correction_mwpm.astype(np.uint8)
        )
        residual_z = (self.h_z @ corrected_data.T % 2).T  # (N, num_z_stabs)

        needs_ml = residual_z.any(axis=1)
        ml_count = int(needs_ml.sum())

        print(f"        [Hybrid] MWPM solved: {N - ml_count}/{N}, "
              f"ML needed: {ml_count}/{N}")

        if ml_count == 0:
            return correction_mwpm

        ml_indices = np.where(needs_ml)[0]
        n_ml = ml_count

        # Stage 2.5: ML 입력 구성 (마지막 라운드의 Z-stab 슬롯에 residual 주입)
        stim_detectors = np.zeros(
            (n_ml, self.num_rounds * self.num_stabilizers), dtype=np.float32
        )
        last_offset = (self.num_rounds - 1) * self.num_stabilizers
        z_start = last_offset + self.residual_offset_in_round
        stim_detectors[:, z_start : z_start + self.num_z_stabs] = (
            residual_z[ml_indices].astype(np.float32)
        )

        # Stage 3: ML inference (mapper 직접 호출)
        if self.model_type == "graph":
            node_features = self.converter.graph_mapper.map_to_node_features(
                stim_detectors
            )
            correction_ml = self.ml_decoder.decode(
                node_features, edge_index=self.converter.edge_index
            )
        else:
            images = self.converter.image_mapper.map_to_images(stim_detectors)
            correction_ml = self.ml_decoder.decode(images)

        if correction_ml.shape[1] > self.num_data:
            correction_ml = correction_ml[:, : self.num_data]

        # 합성
        combined = correction_mwpm.copy()
        combined[ml_indices] = (
            correction_mwpm[ml_indices].astype(np.uint8)
            ^ correction_ml.astype(np.uint8)
        ).astype(np.int8)

        return combined
