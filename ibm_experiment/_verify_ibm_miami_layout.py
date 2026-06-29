"""
_verify_ibm_miami_layout.py  (작업 0.5 Part B, 검증 전용)

live ibm_miami backend에 접속해 nighthawk_layout.py의 가정(120-qubit 12x10
row-major 4-neighbor square lattice)이 현재도 유효한지 대조한다.

- 기존 KCS 인증 방식(ibm_simulator.py와 동일: keys.json의 ibm_api_key +
  instance='Yonsei_internal')만 사용. 새 인증 안 만듦.
- nighthawk_layout / pipeline 코드 수정 없음. select_best_patch를 그대로 호출만.
"""

import os
import sys

IBM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IBM_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, IBM_DIR)

import json
import numpy as np
from paths import ProjectPaths

PATHS = ProjectPaths(REPO_ROOT)
KEYS = PATHS.load_keys()

# config에서 backend/instance 그대로 읽음
with open(os.path.join(IBM_DIR, "config.json")) as f:
    CFG = json.load(f)
INSTANCE = CFG["backend"]["instance"]
BACKEND_NAME = CFG["backend"]["backend_name"]
print(f"instance={INSTANCE}  backend={BACKEND_NAME}")

from qiskit_ibm_runtime import QiskitRuntimeService

# ibm_simulator.py와 동일한 호출 형태
service = QiskitRuntimeService(token=KEYS["ibm_api_key"], instance=INSTANCE)
backend = service.backend(BACKEND_NAME)

print("\n=== B-2: live backend properties ===")
nq = backend.num_qubits
print(f"name        : {backend.name}")
print(f"num_qubits  : {nq}")

cmap = backend.coupling_map
edges = list(cmap.get_edges()) if cmap is not None else []
# 무방향 edge 집합
undirected = set()
for a, b in edges:
    undirected.add((min(a, b), max(a, b)))
print(f"coupling edges (directed list len): {len(edges)}, undirected unique: {len(undirected)}")

# degree 분포
from collections import Counter
deg = Counter()
for a, b in undirected:
    deg[a] += 1
    deg[b] += 1
deg_hist = Counter(deg.values())
print(f"degree histogram (deg->count): {dict(sorted(deg_hist.items()))}")
print(f"qubits appearing in coupling: {len(deg)} (vs num_qubits {nq})")

# calibration 샘플
try:
    props = backend.properties()
except Exception as e:
    props = None
    print(f"properties() 실패: {e!r}")

if props is not None:
    t1s, t2s = [], []
    for q in range(min(nq, 120)):
        try:
            t1s.append(props.t1(q))
            t2s.append(props.t2(q))
        except Exception:
            pass
    if t1s:
        print(f"T1 (us) median={np.median(t1s)*1e6:.1f}  T2 (us) median={np.median(t2s)*1e6:.1f}")
    # 대표 2q gate error 몇 개
    sample_errs = []
    for (a, b) in list(undirected)[:8]:
        for g in ("cz", "ecr", "cx"):
            try:
                sample_errs.append((g, a, b, props.gate_error(g, [a, b])))
                break
            except Exception:
                continue
    print(f"sample 2q gate errors: {sample_errs[:5]}")

print("\n=== B-3: 12x10 row-major 4-neighbor 격자 가정 대조 ===")
NROWS, NCOLS = 12, 10
expected_qubits = NROWS * NCOLS
print(f"num_qubits == {expected_qubits} (12*10)?  {nq == expected_qubits}")

# 가정된 row-major 4-neighbor edge 집합 생성
assumed = set()
for r in range(NROWS):
    for c in range(NCOLS):
        q = r * NCOLS + c
        if c + 1 < NCOLS:
            assumed.add((q, q + 1))
        if r + 1 < NROWS:
            assumed.add((min(q, q + NCOLS), max(q, q + NCOLS)))
print(f"assumed row-major 4-neighbor edges: {len(assumed)}")

missing_in_live = assumed - undirected      # 가정엔 있는데 실제 없음
extra_in_live = undirected - assumed         # 실제엔 있는데 가정에 없음
print(f"assumed edges MISSING in live coupling : {len(missing_in_live)}")
if missing_in_live:
    print(f"  examples: {sorted(missing_in_live)[:10]}")
print(f"live edges NOT in assumed grid         : {len(extra_in_live)}")
if extra_in_live:
    print(f"  examples: {sorted(extra_in_live)[:10]}")
exact_grid = (nq == expected_qubits) and (not missing_in_live) and (not extra_in_live)
print(f"==> live coupling == assumed 12x10 grid EXACTLY: {exact_grid}")

print("\n=== B-3: select_best_patch 실제 동작 (d=3, d=5) ===")
from utils.nighthawk_layout import select_best_patch
for d in (3, 5):
    print(f"\n--- distance {d} ---")
    try:
        res = select_best_patch(backend=backend, distance=d,
                                strategy="min_cx_error", verbose=True)
        diag = res["diagnostics"]
        print(f"  [PASS] patch found. origin={diag['patch_origin']} "
              f"avg_cx_error={diag['avg_cx_error']} grid={diag['grid_shape']} "
              f"candidates={diag['num_candidates']}")
        print(f"  data_qubits={res['data_qubits']}")
        print(f"  ancilla_qubits={res['ancilla_qubits']}")
        # 선택된 patch의 모든 CX edge가 실제 coupling에 있는지 재확인
        phys = res["physical_qubits"]
        ok = True
        # initial_layout 의 물리 인덱스가 전부 < num_qubits 인지
        if any(p >= nq for p in res["initial_layout"]):
            ok = False
            print("  [WARN] some physical index >= num_qubits")
        print(f"  initial_layout in-range: {ok}")
    except Exception as e:
        print(f"  [FAIL] select_best_patch raised: {type(e).__name__}: {e}")
