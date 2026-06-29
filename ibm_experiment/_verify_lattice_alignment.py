"""
_verify_lattice_alignment.py  (작업 2a, 검증 전용 — 나중에 정리)

KCS SurfaceCodeCircuit (O1) vs NVIDIA Ising data_mapping/SurfaceCode (XV) 가
동일한 rotated surface code 격자를 가리키는지 numeric으로 대조한다.

- adapter/변환 코드 아님. 읽기/대조/출력만.
- KCS/NVIDIA 소스 수정 없음.
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVIDIA_CODE = os.path.join(REPO_ROOT, "Ising-Decoding", "code")

sys.path.insert(0, REPO_ROOT)        # for ibm_experiment.*
sys.path.insert(0, NVIDIA_CODE)      # for qec.surface_code.*

from ibm_experiment.circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
    construct_X_stab_Parity_check_Mat,
    construct_Z_stab_Parity_check_Mat,
)

try:
    from qec.surface_code.memory_circuit import SurfaceCode
    HAVE_SURFACECODE = True
except Exception as e:  # noqa
    HAVE_SURFACECODE = False
    SURFACECODE_ERR = repr(e)


def supports_from_parity(H):
    """parity matrix (m, n) -> list of frozenset(support data-qubit indices), row order."""
    H = np.asarray(H)
    return [frozenset(np.nonzero(H[i])[0].tolist()) for i in range(H.shape[0])]


def supports_from_lists(stab_lists):
    return [frozenset(s) for s in stab_lists]


def plaquette_phase_grid(x_supports, z_supports, d):
    """(d-1)x(d-1) grid: 'X'/'Z' for each bulk(weight-4) plaquette by its top-left cell, '.' if none."""
    grid = [["."] * (d - 1) for _ in range(d - 1)]
    for typ, sups in (("X", x_supports), ("Z", z_supports)):
        for s in sups:
            if len(s) != 4:
                continue
            tl = min(s)  # top-left data qubit index
            r, c = tl // d, tl % d
            if 0 <= r < d - 1 and 0 <= c < d - 1:
                grid[r][c] = typ
    return grid


def boundary_edges(x_supports, z_supports, d):
    """weight-2 stab을 edge(top/bottom/left/right)로 분류해 type별 집합 반환."""
    def classify(s):
        coords = [(i // d, i % d) for i in s]
        rs = [a for a, _ in coords]
        cs = [b for _, b in coords]
        if rs[0] == rs[1]:        # same row -> horizontal pair -> top/bottom edge
            return "top" if rs[0] == 0 else "bottom"
        else:                      # same col -> vertical pair -> left/right edge
            return "left" if cs[0] == 0 else "right"
    out = {"X": {}, "Z": {}}
    for typ, sups in (("X", x_supports), ("Z", z_supports)):
        for s in sups:
            if len(s) == 2:
                e = classify(s)
                out[typ].setdefault(e, 0)
                out[typ][e] += 1
    return out


def char_grid_from_anchors(anchor_idx_X, anchor_idx_Z, d):
    """각 stab anchor data-qubit 위치에 'X'/'Z' 표기한 d x d 문자 그리드."""
    g = [["."] * d for _ in range(d)]
    for a in anchor_idx_X:
        g[a // d][a % d] = "X"
    for a in anchor_idx_Z:
        a = int(a)
        if g[a // d][a % d] == "X":
            g[a // d][a % d] = "*"  # collision (should not happen)
        else:
            g[a // d][a % d] = "Z"
    return g


def print_grid(g, indent="    "):
    for row in g:
        print(indent + " ".join(str(x) for x in row))


def fmt_mat(m, indent="    "):
    for row in m:
        print(indent + " ".join(f"{v:>3.1f}" for v in row))


def analyze(d, full=True):
    print("=" * 72)
    print(f"distance d = {d}")
    print("=" * 72)

    # ---------- KCS ----------
    kcs = SurfaceCodeCircuit(distance=d)
    kcs_X = supports_from_lists(kcs.x_stabilizers)
    kcs_Z = supports_from_lists(kcs.z_stabilizers)
    kcs_lx = sorted(kcs.logical_x)
    kcs_lz = sorted(kcs.logical_z)

    # ---------- NVIDIA: pure parity constructors (XV) ----------
    nvX_par = supports_from_parity(construct_X_stab_Parity_check_Mat(d).numpy())
    nvZ_par = supports_from_parity(construct_Z_stab_Parity_check_Mat(d).numpy())

    # ---------- NVIDIA: SurfaceCode(d,'X','V') hx/hz/lx/lz ----------
    nv_sc = None
    if HAVE_SURFACECODE:
        try:
            nv_sc = SurfaceCode(d, first_bulk_syndrome_type='X', rotated_type='V')
        except Exception as e:  # noqa
            print(f"[SurfaceCode 인스턴스화 실패: {e!r}]")
            nv_sc = None

    print(f"\n[개수] KCS: |X|={len(kcs_X)} |Z|={len(kcs_Z)}  "
          f"NVIDIA(parity): |X|={len(nvX_par)} |Z|={len(nvZ_par)}  "
          f"(이론 각 (d^2-1)/2 = {(d*d-1)//2})")

    # ===== Step1/2 표 (d=3 전체) =====
    if full:
        def stab_table(name, sups, d):
            print(f"\n--- {name} (support: idx -> (row,col), weight, bulk/boundary) ---")
            for i, s in enumerate(sorted(sups, key=lambda x: min(x))):
                cells = sorted(s)
                rc = [(c // d, c % d) for c in cells]
                kind = "bulk" if len(s) == 4 else "bndry"
                print(f"  [{i}] {cells}  {rc}  w={len(s)} {kind}")
        print("\n### KCS")
        stab_table("KCS X-stabs", kcs_X, d)
        stab_table("KCS Z-stabs", kcs_Z, d)
        print("\n### NVIDIA (construct_*_Parity, XV)")
        stab_table("NV X-stabs", nvX_par, d)
        stab_table("NV Z-stabs", nvZ_par, d)

    # ===== 3-1: X/Z 역할 + boundary =====
    print("\n----- 3-1. X/Z role + boundary -----")
    set_kcs_X, set_kcs_Z = set(kcs_X), set(kcs_Z)
    set_nv_X, set_nv_Z = set(nvX_par), set(nvZ_par)

    same_direct = (set_kcs_X == set_nv_X) and (set_kcs_Z == set_nv_Z)
    same_swapped = (set_kcs_X == set_nv_Z) and (set_kcs_Z == set_nv_X)
    print(f"  KCS X-supports == NVIDIA X-supports & KCS Z == NVIDIA Z : {same_direct}")
    print(f"  KCS X-supports == NVIDIA Z-supports & KCS Z == NVIDIA X : {same_swapped}  (X<->Z swap)")
    union_same = (set_kcs_X | set_kcs_Z) == (set_nv_X | set_nv_Z)
    print(f"  전체 stabilizer support 집합(타입 무시) 동일 : {union_same}")

    be_kcs = boundary_edges(kcs_X, kcs_Z, d)
    be_nv = boundary_edges(nvX_par, nvZ_par, d)
    print(f"  KCS    boundary edges: X={be_kcs['X']}  Z={be_kcs['Z']}")
    print(f"  NVIDIA boundary edges: X={be_nv['X']}  Z={be_nv['Z']}")

    # anchor char grid (data_mapping 기반)
    aX = compute_stabX_to_data_index_map(d, 'XV')
    aZ = compute_stabZ_to_data_index_map(d, 'XV')
    aX = [int(v) for v in (aX.tolist() if hasattr(aX, 'tolist') else aX)]
    aZ = [int(v) for v in (aZ.tolist() if hasattr(aZ, 'tolist') else aZ)]
    print("\n  NVIDIA anchor char-grid (data_mapping XV, 'X'/'Z'/'.'):")
    print_grid(char_grid_from_anchors(aX, aZ, d))

    # ===== 3-2: bulk checkerboard 위상 =====
    print("\n----- 3-2. bulk checkerboard phase -----")
    g_kcs = plaquette_phase_grid(kcs_X, kcs_Z, d)
    g_nv = plaquette_phase_grid(nvX_par, nvZ_par, d)
    print("  KCS bulk plaquette grid ((d-1)x(d-1), top-left cell):")
    print_grid(g_kcs)
    print("  NVIDIA bulk plaquette grid:")
    print_grid(g_nv)
    tl_kcs = g_kcs[0][0]
    tl_nv = g_nv[0][0]
    print(f"  top-left bulk plaquette: KCS={tl_kcs}, NVIDIA={tl_nv}, 일치={tl_kcs == tl_nv}")
    mism = [(r, c, g_kcs[r][c], g_nv[r][c])
            for r in range(d - 1) for c in range(d - 1) if g_kcs[r][c] != g_nv[r][c]]
    print(f"  불일치 plaquette 수: {len(mism)}" + (f"  {mism}" if mism else ""))

    # ===== 3-3: logical operators =====
    print("\n----- 3-3. logical operators -----")
    # NVIDIA EvalModule 방식 (logical_error_rate.py:573-578, XV/ZH): Lx=top row, Lz=left col
    nv_eval_Lx = sorted(range(d))                    # [0..d-1]
    nv_eval_Lz = sorted(range(0, d * d, d))          # left column
    print(f"  KCS logical_x        : {kcs_lx}")
    print(f"  NVIDIA EvalModule Lx : {nv_eval_Lx}   일치={kcs_lx == nv_eval_Lx}")
    print(f"  KCS logical_z        : {kcs_lz}")
    print(f"  NVIDIA EvalModule Lz : {nv_eval_Lz}   일치={kcs_lz == nv_eval_Lz}")
    if nv_sc is not None:
        sc_lx = sorted(np.nonzero(np.asarray(nv_sc.lx).flatten())[0].tolist())
        sc_lz = sorted(np.nonzero(np.asarray(nv_sc.lz).flatten())[0].tolist())
        print(f"  NVIDIA SurfaceCode.lx: {sc_lx}   KCS와 일치={kcs_lx == sc_lx}")
        print(f"  NVIDIA SurfaceCode.lz: {sc_lz}   KCS와 일치={kcs_lz == sc_lz}")
        # hx/hz support 대조
        sc_X = set(supports_from_parity(nv_sc.hx))
        sc_Z = set(supports_from_parity(nv_sc.hz))
        print(f"  SurfaceCode.hx supports == parity-constructor X : {sc_X == set_nv_X}")
        print(f"  SurfaceCode.hz supports == parity-constructor Z : {sc_Z == set_nv_Z}")
        print(f"  SurfaceCode.hx == KCS X : {sc_X == set_kcs_X} | == KCS Z (swap): {sc_X == set_kcs_Z}")
    else:
        print(f"  [SurfaceCode 미사용: {SURFACECODE_ERR if not HAVE_SURFACECODE else 'instantiation failed'}]")

    # ===== 3-4: presence map =====
    print("\n----- 3-4. presence map (boundary 0.5 / bulk 1.0) -----")
    wX = normalized_weight_mapping_Xstab_memory(d, 'XV').reshape(d, d).numpy()
    wZ = normalized_weight_mapping_Zstab_memory(d, 'XV').reshape(d, d).numpy()

    # KCS supports로부터 동일 anchor 규칙을 적용해 presence 재구성
    def kcs_presence(x_sups, z_sups, d):
        pX = np.zeros((d, d)); pZ = np.zeros((d, d))
        # X anchor 규칙(NVIDIA XV doc): bulk->top-left(min row,min col);
        #   boundary horizontal(top/bottom)->LEFT(min col); vertical(left/right)->TOP(min row)
        def anchor_X(s):
            cells = [(i // d, i % d, i) for i in s]
            if len(s) == 2:
                rs = [a for a, _, _ in cells]
                if rs[0] == rs[1]:
                    cells.sort(key=lambda x: x[1])      # LEFT
                else:
                    cells.sort(key=lambda x: x[0])      # TOP
            else:
                cells.sort(key=lambda x: (x[0], x[1]))  # top-left
            return cells[0][0], cells[0][1]
        # Z anchor 규칙(NVIDIA XV doc): bulk->top-right(min row,max col);
        #   boundary vertical->TOP(min row); horizontal->RIGHT(max col)
        def anchor_Z(s):
            cells = [(i // d, i % d, i) for i in s]
            if len(s) == 2:
                cs = [c for _, c, _ in cells]
                if cs[0] == cs[1]:
                    cells.sort(key=lambda x: x[0])       # TOP
                else:
                    cells.sort(key=lambda x: -x[1])      # RIGHT
            else:
                cells.sort(key=lambda x: (x[0], -x[1]))  # top-right
            return cells[0][0], cells[0][1]
        for s in x_sups:
            r, c = anchor_X(s); pX[r, c] = 0.5 if len(s) == 2 else 1.0
        for s in z_sups:
            r, c = anchor_Z(s); pZ[r, c] = 0.5 if len(s) == 2 else 1.0
        return pX, pZ

    kpX, kpZ = kcs_presence(kcs_X, kcs_Z, d)
    print("  NVIDIA X-presence:");  fmt_mat(wX)
    print("  KCS-derived X-presence:");  fmt_mat(kpX)
    print(f"  X-presence 동일: {np.array_equal(wX, kpX)}")
    if not np.array_equal(wX, kpX):
        print("   diff (NVIDIA-KCS):"); fmt_mat(wX - kpX)
    print("  NVIDIA Z-presence:");  fmt_mat(wZ)
    print("  KCS-derived Z-presence:");  fmt_mat(kpZ)
    print(f"  Z-presence 동일: {np.array_equal(wZ, kpZ)}")
    if not np.array_equal(wZ, kpZ):
        print("   diff (NVIDIA-KCS):"); fmt_mat(wZ - kpZ)

    return {
        "same_direct": same_direct, "same_swapped": same_swapped, "union_same": union_same,
        "tl_match": tl_kcs == tl_nv, "bulk_mismatch": len(mism),
        "lx_match": kcs_lx == nv_eval_Lx, "lz_match": kcs_lz == nv_eval_Lz,
        "xpres_match": bool(np.array_equal(wX, kpX)), "zpres_match": bool(np.array_equal(wZ, kpZ)),
    }


if __name__ == "__main__":
    print(f"HAVE_SURFACECODE = {HAVE_SURFACECODE}")
    res3 = analyze(3, full=True)
    res5 = analyze(5, full=False)
    print("\n" + "#" * 72)
    print("SUMMARY")
    print("#" * 72)
    for d, r in (("d=3", res3), ("d=5", res5)):
        print(f"{d}: {r}")
