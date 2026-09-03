"""
NVIDIA Ising pre-decoder wrapper for KCS pipeline (Step 1).

작업 1.5/2a 사전 확정:
- 입력 텐서: (B, 4, T, D, D) float32, channel-first
    ch0: x_type      X-stab detection events grid (per-round)
    ch1: z_type      Z-stab detection events grid
    ch2: x_present   X-stab presence/weight map (boundary 0.5, bulk 1.0)
    ch3: z_present   Z-stab presence/weight map
- KCS는 Z-basis memory → datapipe의 basis='Z' 마스킹과 일치하게
    x_syn_diff[round=0, last]=0, x_present[round=0, last]=0  (X가 off-basis)
- KCS ↔ NVIDIA-XV 격자: support·logical·presence 셀단위 일치 (작업 2a 검증), X/Z swap 없음.
- 단 stab enumeration 순서는 다름 → index 매핑 대신 grid 좌표 기반 매핑(작업 2a 권장).
- ibm_miami(Nighthawk)는 ancilla reuse → raw 가 누적 XOR → single-differencing 한 단계 추가.

본 파일은 wrapper 3개 함수만 제공 (Step 1):
  load_nvidia_predecoder(weight_path, distance, n_rounds, device, model_id=4) -> model
  kcs_to_nvidia_input(syndromes, data_states, x_stabilizers, z_stabilizers,
                       distance, n_rounds, basis='Z', device='cuda') -> Tensor
  run_nvidia_predecoder(model, kcs_input) -> Tensor (B, 4, T, D, D)

Step 2(잔차 디코드 + correction frame)와 Step 3(평가 루프 통합)은 후속.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import torch


# -----------------------------------------------------------------------------
# NVIDIA repo sys.path 등록 (idempotent)
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NVIDIA_CODE = os.path.join(_REPO_ROOT, "Ising-decoding", "code")
if _NVIDIA_CODE not in sys.path:
    sys.path.insert(0, _NVIDIA_CODE)


# -----------------------------------------------------------------------------
# NVIDIA hidden defaults (from Ising-decoding/code/workflows/config_validator.py)
# -----------------------------------------------------------------------------
_NVIDIA_DROPOUT_P = 0.05
_NVIDIA_ACTIVATION = "gelu"


def load_nvidia_predecoder(weight_path: str,
                            distance: int,
                            n_rounds: int,
                            device: torch.device,
                            model_id: int = 4):
    """
    NVIDIA PreDecoderModelMemory_v1 인스턴스화 + .pt weight 로드 + eval 모드.

    Args:
        weight_path: Accurate(.pt) 또는 Fast(.pt) 경로
        distance, n_rounds: cfg에 전달 (model.forward는 (B,4,T,D,D) 어떤 D/T든 OK이지만
                            cfg 객체에 distance/n_rounds 속성이 필요)
        device: KC2 torch device (예: torch.device('cuda'))
        model_id: NVIDIA registry id (Accurate=4, Fast=1). default 4.

    Returns:
        torch.nn.Module on `device`, .eval() 모드.
    """
    from model.predecoder import PreDecoderModelMemory_v1
    from model.registry import get_model_spec

    spec = get_model_spec(model_id)

    cfg = SimpleNamespace()
    cfg.distance = int(distance)
    cfg.n_rounds = int(n_rounds)
    cfg.model = SimpleNamespace()
    cfg.model.dropout_p = _NVIDIA_DROPOUT_P
    cfg.model.activation = _NVIDIA_ACTIVATION
    cfg.model.input_channels = 4
    cfg.model.out_channels = int(spec.num_filters[-1])  # Accurate=4, Fast=4
    cfg.model.num_filters = list(spec.num_filters)
    cfg.model.kernel_size = list(spec.kernel_size)

    model = PreDecoderModelMemory_v1(cfg)

    raw = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw).__name__}")
    # Strip 'module.' prefix (DDP-wrapped checkpoints)
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v
                  for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch — missing={missing}, unexpected={unexpected}")

    model.to(device).eval()
    return model


# -----------------------------------------------------------------------------
# Anchor 규칙 (NVIDIA data_mapping.py 문서 + 작업 2a 검증)
# -----------------------------------------------------------------------------
def _anchor_X(support: Sequence[int], d: int):
    """X-stab support → grid anchor (row, col).

    Rule (per data_mapping.py docstring):
      - bulk(weight-4)         → top-left (smallest row, then smallest col)
      - boundary horizontal(top/bottom) → LEFT (smallest col)
      - boundary vertical(left/right)   → TOP  (smallest row)
    """
    cells = [(i // d, i % d) for i in support]
    if len(cells) == 2:
        rs = [r for r, _ in cells]
        if rs[0] == rs[1]:           # horizontal pair
            cells.sort(key=lambda x: x[1])         # LEFT
        else:                         # vertical pair
            cells.sort(key=lambda x: x[0])         # TOP
    else:                             # bulk
        cells.sort(key=lambda x: (x[0], x[1]))     # top-left
    return cells[0]


def _anchor_Z(support: Sequence[int], d: int):
    """Z-stab support → grid anchor (row, col).

    Rule:
      - bulk(weight-4)         → top-right (smallest row, then largest col)
      - boundary vertical(left/right)   → TOP   (smallest row)
      - boundary horizontal(top/bottom) → RIGHT (largest col)
    """
    cells = [(i // d, i % d) for i in support]
    if len(cells) == 2:
        cs = [c for _, c in cells]
        if cs[0] == cs[1]:           # vertical pair
            cells.sort(key=lambda x: x[0])         # TOP
        else:                         # horizontal pair
            cells.sort(key=lambda x: -x[1])        # RIGHT
    else:                             # bulk
        cells.sort(key=lambda x: (x[0], -x[1]))    # top-right
    return cells[0]


# -----------------------------------------------------------------------------
# 입력 변환: KCS HW syndromes → NVIDIA (B, 4, T, D, D)
# -----------------------------------------------------------------------------
def _cumulative_to_per_round(syndromes_np: np.ndarray) -> np.ndarray:
    """raw_r ⊕ raw_{r-1} = s_r (raw_{-1}=0). ancilla-reuse 회로의 누적 XOR을 푼다.

    동일 로직이 ibm_experiment/extractors/stim_compat.py:_cumulative_to_per_round 에도 있음
    (재사용을 위해 import해도 되지만 의존성 최소화 위해 여기에 inline).
    """
    s = syndromes_np.astype(np.int8, copy=False)
    out = np.zeros_like(s)
    out[:, 0, :] = s[:, 0, :]
    if s.shape[1] > 1:
        out[:, 1:, :] = (s[:, 1:, :] ^ s[:, :-1, :])
    return out


def kcs_to_nvidia_input(
    syndromes: np.ndarray,                # (N, T, num_stab) raw HW (KCS 순서)
    data_states: np.ndarray,              # (N, num_data) — Step 1엔 미사용, 인터페이스 보존
    x_stabilizers: List[List[int]],       # KCS SurfaceCodeCircuit.x_stabilizers
    z_stabilizers: List[List[int]],       # KCS SurfaceCodeCircuit.z_stabilizers
    distance: int,
    n_rounds: int,
    basis: str = "Z",                     # KCS 회로 = Z-basis memory
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    KCS 입력 → NVIDIA (B, 4, T, D, D) float32.

    파이프라인:
      1) cumulative→per_round (ancilla reuse single-differencing)
      2) [X-stab cols | Z-stab cols] 로 분리 (KCS는 X-stabs 먼저)
      3) Round간 XOR diff with zero prepend → detection events
      4) basis='Z'에서 X-stab(off-basis) 첫·마지막 round 0 마스킹
      5) 각 stab의 support → NVIDIA anchor 규칙으로 grid (row, col) 산출
         (작업 2a 검증: KCS support == NVIDIA support, X/Z swap 없음. 순서는 다름 → 좌표 매핑.)
      6) presence map (boundary 0.5 / bulk 1.0) — X용/Z용; basis='Z'에서 X presence도 round0,-1 0
      7) 4 채널 stack → (N, 4, T, D, D) float32
    """
    basis_u = str(basis).upper()
    if basis_u not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

    N, T, num_stab = syndromes.shape
    if T != int(n_rounds):
        raise ValueError(f"syndromes T={T} != n_rounds={n_rounds}")
    d = int(distance)
    n_x = len(x_stabilizers)
    n_z = len(z_stabilizers)
    if n_x + n_z != num_stab:
        raise ValueError(f"len(x)+len(z)={n_x+n_z} != syndromes num_stab={num_stab}")

    # 1) cumulative -> per_round (ancilla reuse)
    per_round = _cumulative_to_per_round(syndromes)                     # (N, T, num_stab) int8

    # 2) split (KCS 순서: X 먼저, Z 나중)
    x_raw = per_round[:, :, :n_x].astype(np.int8, copy=False)           # (N, T, n_x)
    z_raw = per_round[:, :, n_x:].astype(np.int8, copy=False)           # (N, T, n_z)

    # 3) XOR diff with zero prepend (per-stab timeline)
    zero_x = np.zeros((N, 1, n_x), dtype=np.int8)
    zero_z = np.zeros((N, 1, n_z), dtype=np.int8)
    x_aug = np.concatenate([zero_x, x_raw], axis=1)                     # (N, T+1, n_x)
    z_aug = np.concatenate([zero_z, z_raw], axis=1)
    x_syn_diff = (x_aug[:, 1:, :] ^ x_aug[:, :-1, :]).astype(np.int8)   # (N, T, n_x)
    z_syn_diff = (z_aug[:, 1:, :] ^ z_aug[:, :-1, :]).astype(np.int8)   # (N, T, n_z)

    # 4) basis='Z' off-basis 마스킹 (X-stab은 round0, last에서 비결정적)
    if basis_u == "Z":
        x_syn_diff[:, 0, :] = 0
        x_syn_diff[:, -1, :] = 0
    else:  # 'X'
        z_syn_diff[:, 0, :] = 0
        z_syn_diff[:, -1, :] = 0

    # 5) grid scatter via anchor rule (KCS support → NVIDIA anchor)
    x_grid = np.zeros((N, T, d, d), dtype=np.float32)
    z_grid = np.zeros((N, T, d, d), dtype=np.float32)
    for i, sup in enumerate(x_stabilizers):
        r, c = _anchor_X(sup, d)
        x_grid[:, :, r, c] = x_syn_diff[:, :, i].astype(np.float32)
    for j, sup in enumerate(z_stabilizers):
        r, c = _anchor_Z(sup, d)
        z_grid[:, :, r, c] = z_syn_diff[:, :, j].astype(np.float32)

    # 6) presence map (D×D) — 같은 anchor 규칙으로 KCS support에서 직접 산출 (작업 2a 동일)
    x_pres = np.zeros((d, d), dtype=np.float32)
    z_pres = np.zeros((d, d), dtype=np.float32)
    for sup in x_stabilizers:
        r, c = _anchor_X(sup, d)
        x_pres[r, c] = 0.5 if len(sup) == 2 else 1.0
    for sup in z_stabilizers:
        r, c = _anchor_Z(sup, d)
        z_pres[r, c] = 0.5 if len(sup) == 2 else 1.0

    # broadcast to (N, T, d, d) + off-basis boundary 마스킹
    x_pres_NTDD = np.broadcast_to(x_pres[None, None, :, :], (N, T, d, d)).copy()
    z_pres_NTDD = np.broadcast_to(z_pres[None, None, :, :], (N, T, d, d)).copy()
    if basis_u == "Z":
        x_pres_NTDD[:, 0, :, :] = 0
        x_pres_NTDD[:, -1, :, :] = 0
    else:
        z_pres_NTDD[:, 0, :, :] = 0
        z_pres_NTDD[:, -1, :, :] = 0

    # 7) (N, 4, T, D, D) stack
    trainX_np = np.stack([x_grid, z_grid, x_pres_NTDD, z_pres_NTDD], axis=1).astype(np.float32)
    trainX = torch.from_numpy(trainX_np).to(device=device, dtype=torch.float32).contiguous()
    return trainX


@torch.no_grad()
def run_nvidia_predecoder(model: torch.nn.Module, kcs_input: torch.Tensor) -> torch.Tensor:
    """forward only. Returns (B, 4, T, D, D)."""
    return model(kcs_input)


# =============================================================================
# Step 2: 잔차 디코드 + correction frame (A1 분기)
# =============================================================================
# NVIDIA EvalModule(logical_error_rate.py:700-789) residual 공식을 KCS stab 순서로
# 재현. 핵심 통찰:
#   residual_detector = baseline_detector(KCS DEM, 검증됨) XOR NN_detector_correction
# 여기서 baseline은 KCS converter.hw_to_mwpm_detectors가 만든 detection events이고,
# NN correction은 NVIDIA가 예측한 syndrome correction(syn_*_grid) + data correction이
# 유발하는 syndrome(S = H @ data_corr)을 detection-event space로 환산한 것.
#
# NVIDIA R 공식 (basis 무관, per stab-type):
#   R[t=0]   = syn_diff[0] + syn_pred[0]               + S[0]
#   R[t>=1]  = syn_diff[t] + syn_pred[t] + syn_pred[t-1] + S[t]
# baseline_detector 가 이미 syn_diff(+boundary)를 담고 있으므로,
#   NN_correction[t=0]  = syn_pred[0]               + S[0]
#   NN_correction[t>=1] = syn_pred[t] + syn_pred[t-1] + S[t]
# final boundary detector 는 NN 이 건드리지 않음(0).
#
# KCS DEM detector 순서 (build_qiskit_style_stim_circuit):
#   round0: Z[0..n_z-1]
#   round r=1..R-1: [X[0..n_x-1], Z[0..n_z-1]]
#   final: Z-boundary[0..n_z-1]
# NVIDIA basis=Z residual layout과 1:1 대응 (half = n_x = n_z).
#
# 검증 게이트: NVIDIA logits를 음수로 강제 → 모든 sample_predictions=0 → NN_correction=0,
# pre_L=0 → correction frame 이 KCS MWPMDecoder.decode_batch(baseline)과 bit-exact 일치.


def _grid_to_stab(grid_NTDD: np.ndarray, stabilizers: List[List[int]], d: int, anchor_fn):
    """(N, T, D, D) grid → (N, n_stab, T) per-stab value (KCS anchor 좌표에서 추출)."""
    N, T = grid_NTDD.shape[0], grid_NTDD.shape[1]
    out = np.zeros((N, len(stabilizers), T), dtype=np.int8)
    for k, sup in enumerate(stabilizers):
        r, c = anchor_fn(sup, d)
        out[:, k, :] = grid_NTDD[:, :, r, c].astype(np.int8)
    return out


def _induced_syndrome(data_corr_flat: np.ndarray, stabilizers: List[List[int]]) -> np.ndarray:
    """data correction이 유발하는 syndrome S[k,t] = XOR_{q in support_k} data_corr[q, t].

    data_corr_flat: (N, D2, T) (row-major data qubit idx). returns (N, n_stab, T)."""
    N, _, T = data_corr_flat.shape
    out = np.zeros((N, len(stabilizers), T), dtype=np.int8)
    for k, sup in enumerate(stabilizers):
        acc = np.zeros((N, T), dtype=np.int8)
        for q in sup:
            acc ^= data_corr_flat[:, q, :].astype(np.int8)
        out[:, k, :] = acc
    return out


def nvidia_to_correction_frame(
    nvidia_out: torch.Tensor,                 # (N, 4, T, D, D) — run_nvidia_predecoder 출력 (raw logits)
    baseline_detectors: np.ndarray,           # (N, num_det) uint8 — KCS converter.hw_to_mwpm_detectors
    matcher,                                  # pymatching.Matching (KCS DEM)
    x_stabilizers: List[List[int]],
    z_stabilizers: List[List[int]],
    distance: int,
    num_rounds: int,
    logical_z: List[int],
    num_data: int,
    th_data: float = 0.0,
    th_syn: float = 0.0,
    verbose_stats: bool = False,
    stats_tag: str = "",
) -> np.ndarray:
    """
    A1: NVIDIA 4채널 출력 → 잔차 detection events → KCS PyMatching → correction frame.

    Returns:
        correction frame (N, num_data) int8. logical_z[0] qubit에 최종 logical-Z flip
        (= NVIDIA pre_L XOR PyMatching pred_obs)을 인코딩. KCS LogicalErrorRateEvaluator가
        data_states ^ correction의 logical_z parity로 LER 계산.

    NOTE: basis='Z' (KCS는 Z-basis memory) 경로만 구현.
    """
    d = int(distance)
    T = int(num_rounds)
    n_x = len(x_stabilizers)
    n_z = len(z_stabilizers)
    n_stab = n_x + n_z

    out = nvidia_out.detach().to("cpu")
    # threshold (sample_predictions, mode='threshold': logits >= th)
    z_data_corr = (out[:, 0] >= th_data).to(torch.int32).numpy()   # (N, T, D, D)
    x_data_corr = (out[:, 1] >= th_data).to(torch.int32).numpy()
    syn_x_grid = (out[:, 2] >= th_syn).to(torch.int32).numpy()
    syn_z_grid = (out[:, 3] >= th_syn).to(torch.int32).numpy()
    N = z_data_corr.shape[0]

    # data correction을 flat (N, D2, T) — grid (N,T,D,D) → (N,D,D,T) → (N,D2,T) row-major
    z_corr_flat = z_data_corr.transpose(0, 2, 3, 1).reshape(N, d * d, T)
    x_corr_flat = x_data_corr.transpose(0, 2, 3, 1).reshape(N, d * d, T)

    # grid → KCS stab 순서 syndrome correction
    syn_x_pred = _grid_to_stab(syn_x_grid, x_stabilizers, d, _anchor_X)   # (N, n_x, T)
    syn_z_pred = _grid_to_stab(syn_z_grid, z_stabilizers, d, _anchor_Z)   # (N, n_z, T)

    # induced syndrome: S_X = Hx @ z_data_corr (X-stab은 Z-error 감지), S_Z = Hz @ x_data_corr
    S_X = _induced_syndrome(z_corr_flat, x_stabilizers)   # (N, n_x, T)
    S_Z = _induced_syndrome(x_corr_flat, z_stabilizers)   # (N, n_z, T)

    # NN correction (detection-event space), per stab-type
    def _nn_corr(syn_pred, S):
        nn = np.zeros_like(syn_pred)
        nn[:, :, 0] = (syn_pred[:, :, 0] + S[:, :, 0]) & 1
        if T > 1:
            nn[:, :, 1:] = (syn_pred[:, :, 1:] + syn_pred[:, :, :-1] + S[:, :, 1:]) & 1
        return nn

    nnX = _nn_corr(syn_x_pred, S_X)   # (N, n_x, T)
    nnZ = _nn_corr(syn_z_pred, S_Z)   # (N, n_z, T)

    # KCS DEM 순서로 배열
    num_det = baseline_detectors.shape[1]
    expected = 2 * n_z + (T - 1) * n_stab
    if num_det != expected:
        raise ValueError(
            f"baseline_detectors width {num_det} != expected {expected} "
            f"(2*n_z + (T-1)*n_stab; n_x={n_x}, n_z={n_z}, T={T})."
        )
    corr_det = np.zeros((N, num_det), dtype=np.uint8)
    # round0: Z[0..n_z-1]
    corr_det[:, 0:n_z] = nnZ[:, :, 0].astype(np.uint8)
    # round r=1..T-1: [X, Z]
    for r in range(1, T):
        base = n_z + (r - 1) * n_stab
        corr_det[:, base:base + n_x] = nnX[:, :, r].astype(np.uint8)
        corr_det[:, base + n_x:base + n_stab] = nnZ[:, :, r].astype(np.uint8)
    # final boundary (마지막 n_z): NN 안 건드림 → 0

    residual = (baseline_detectors.astype(np.uint8) ^ corr_det)

    if verbose_stats:
        base = baseline_detectors.astype(np.uint8)
        b_w = base.sum(axis=1)
        r_w = residual.sum(axis=1)
        b_mean_w = float(b_w.mean())
        r_mean_w = float(r_w.mean())
        reduction = (b_mean_w / r_mean_w) if r_mean_w > 0 else float("inf")
        print(f"[NVIDIA residual stats] {stats_tag}")
        print(f"  baseline detector: mean_nonzero={float(base.mean()):.4f}, "
              f"mean_weight={b_mean_w:.3f}, max_weight={int(b_w.max())}")
        print(f"  residual detector: mean_nonzero={float(residual.mean()):.4f}, "
              f"mean_weight={r_mean_w:.3f}, max_weight={int(r_w.max())}")
        print(f"  syndrome density reduction: {reduction:.2f}x  "
              f"(N_unique={residual.shape[0]}, num_det={residual.shape[1]})")

    # PyMatching decode (KCS DEM) → logical-Z observable 예측
    pred = matcher.decode_batch(residual)
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    pred_obs = pred[:, 0].astype(np.int8)

    # NVIDIA 직접 logical 예측 (basis=Z): logical_z support × all rounds 의 x_data_corr XOR
    pre_L = np.zeros(N, dtype=np.int8)
    for q in logical_z:
        pre_L ^= (x_corr_flat[:, q, :].sum(axis=1) & 1).astype(np.int8)

    final_L = (pred_obs ^ pre_L).astype(np.int8)

    correction = np.zeros((N, num_data), dtype=np.int8)
    correction[:, logical_z[0]] = final_L
    return correction
