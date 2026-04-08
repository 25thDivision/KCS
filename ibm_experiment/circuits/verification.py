# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict
# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# degrees = {q: len(adj[q]) for q in adj}
# vertices = sorted([q for q, d in degrees.items() if d == 3])

# # Heavy-hex에서 degree-3 vertex 간 경로 찾기
# # vertex - link - vertex 패턴
# print("=== Vertex-to-Vertex connections (via link qubits) ===")
# for v in vertices:
#     for link in sorted(adj[v]):
#         if degrees[link] == 2:
#             # link의 다른 쪽 끝
#             other = [q for q in adj[link] if q != v]
#             if other:
#                 o = other[0]
#                 if degrees.get(o, 0) == 3:
#                     if o > v:  # 중복 방지
#                         print(f"  vertex {v} --[link {link}]-- vertex {o}")

# # d=3 (3,3) surface code에 필요한 큐빗 수
# print(f"\n=== d=3 surface code needs ===")
# print(f"  Data qubits: 9")
# print(f"  Ancilla qubits: 8")
# print(f"  Bridge qubits: ~4-8 (topology dependent)")
# print(f"  Total: ~21-25 qubits")

# # 중심부 근처 vertex 클러스터 찾기
# print(f"\n=== Candidate region (vertices 39-68) ===")
# for v in vertices:
#     if 39 <= v <= 68:
#         v_neighbors_via_link = []
#         for link in sorted(adj[v]):
#             if degrees[link] == 2:
#                 other = [q for q in adj[link] if q != v]
#                 if other and degrees.get(other[0], 0) == 3:
#                     v_neighbors_via_link.append((link, other[0]))
#         print(f"  vertex {v}: links to vertices {v_neighbors_via_link}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict
# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# degrees = {q: len(adj[q]) for q in adj}

# # 중심부 영역의 전체 연결 구조 출력
# # vertices 41,43,45 / 60,62,64 근처가 (3,3) 후보
# region = set()
# for v in [41, 43, 45, 60, 62, 64]:
#     region.add(v)
#     for n in adj[v]:
#         region.add(n)
#         for nn in adj[n]:
#             region.add(nn)

# print("=== Region around vertices 41-64 ===")
# for q in sorted(region):
#     neighbors = sorted(adj[q])
#     deg = degrees[q]
#     label = "V" if deg == 3 else "L" if deg == 2 else "B"S
#     print(f"  q{q:3d} [deg={deg},{label}]: neighbors={neighbors}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict
# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# degrees = {q: len(adj[q]) for q in adj}

# # 3x5 영역 필요 — vertices 3행 이상 필요
# # Row 1: 20,22,24,26,28,30
# # Row 2: 39,41,43,45,47,49
# # Row 3: 58,60,62,64,66,68
# # 이 중 3x3 또는 3x5 영역

# # 논문의 (3,3) code는 vertex + link + bridge로 구성
# # 전체 연결 구조를 행/열로 정리

# rows = {
#     0: [4, 8, 12],
#     1: [20, 22, 24, 26, 28, 30],
#     2: [39, 41, 43, 45, 47, 49],
#     3: [58, 60, 62, 64, 66, 68],
#     4: [77, 79, 81, 83, 85, 87],
#     5: [96, 98, 100, 102, 104, 106],
# }

# print("=== Heavy-hex vertex grid ===")
# for r, vs in rows.items():
#     print(f"  Row {r}: {vs}")

# # Row 1-3, Col 0-2 영역의 모든 큐빗 (vertex + link)
# region_vertices = [22, 24, 26, 41, 43, 45, 60, 62, 64]
# region_all = set()
# for v in region_vertices:
#     region_all.add(v)
#     for n in adj[v]:
#         region_all.add(n)

# print(f"\n=== 3x3 region (rows 1-3, cols 1-3) ===")
# print(f"Vertices: {sorted([q for q in region_all if degrees.get(q,0)==3])}")
# print(f"Links:    {sorted([q for q in region_all if degrees.get(q,0)==2])}")
# print(f"Total:    {len(region_all)} qubits")

# for q in sorted(region_all):
#     ns = sorted(adj[q])
#     in_region = [n for n in ns if n in region_all]
#     out_region = [n for n in ns if n not in region_all]
#     label = "V" if degrees[q]==3 else "L"
#     print(f"  q{q:3d}[{label}]: in_region={in_region}, outside={out_region}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict
# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# # 양방향 adjacency
# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# # 내가 배치한 큐빗들
# data_qubits = [41, 42, 43, 53, 60, 34, 61, 62, 63]
# x_ancilla = [40, 44, 59, 54]
# z_ancilla = [22, 24, 23, 64]

# x_stabs = [[0,1,3,4], [1,2,4,5], [3,4,6,7], [4,5,7,8]]
# z_stabs = [[0,1], [1,2,4,5], [3,6], [4,5,7,8]]

# print("=== CX Connectivity Check ===")
# print("\nZ stabilizers (CX: data → ancilla):")
# for i, (anc, dlist) in enumerate(zip(z_ancilla, z_stabs)):
#     for d_local in dlist:
#         d_phys = data_qubits[d_local]
#         ok = d_phys in adj[anc]
#         print(f"  Z{i}: CX(q{d_phys} → q{anc}) {'✅' if ok else '❌ NOT ADJACENT'}")

# print("\nX stabilizers (CX: ancilla → data):")
# for i, (anc, dlist) in enumerate(zip(x_ancilla, x_stabs)):
#     for d_local in dlist:
#         d_phys = data_qubits[d_local]
#         ok = d_phys in adj[anc]
#         print(f"  X{i}: CX(q{anc} → q{d_phys}) {'✅' if ok else '❌ NOT ADJACENT'}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit import QuantumCircuit, transpile
# from collections import defaultdict

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')

# # 현재 surface code 회로의 depth 확인
# import sys
# sys.path.append('ibm_experiment')
# sys.path.append('.')
# from ibm_experiment.circuits.qiskit_surface_code_generator import SurfaceCodeCircuit

# sc = SurfaceCodeCircuit(distance=3, num_rounds=3)
# qc = sc.build_circuit(initial_state=0)
# print(f"Original circuit: {qc.num_qubits} qubits, depth={qc.depth()}")

# # optimization_level별 transpile depth 비교
# for opt in [0, 1, 2, 3]:
#     tqc = transpile(qc, backend=backend, optimization_level=opt)
#     cx_count = tqc.count_ops().get('cx', 0) + tqc.count_ops().get('ecr', 0) + tqc.count_ops().get('cz', 0)
#     print(f"  opt_level={opt}: depth={tqc.depth()}, 2q_gates={cx_count}")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# # Data qubit mapping
# D = {0:22, 1:24, 2:26, 3:41, 4:43, 5:45, 6:60, 7:62, 8:64}

# # Our stabilizers (from SurfaceCodeCircuit)
# x_stabs = [[0,1,3,4], [4,5,7,8], [3,6], [2,5]]
# z_stabs = [[1,2,4,5], [3,4,6,7], [0,1], [7,8]]

# # For each weight-4 stabilizer, find which data-data connections exist
# print("=== Weight-4 stabilizer connectivity ===")
# all_stabs = [("X", s) for s in x_stabs] + [("Z", s) for s in z_stabs]

# for typ, stab in all_stabs:
#     if len(stab) < 4:
#         continue
#     phys = [D[d] for d in stab]
#     print(f"\n{typ} stab {stab} -> physical {phys}")
#     for i in range(len(stab)):
#         for j in range(i+1, len(stab)):
#             p1, p2 = D[stab[i]], D[stab[j]]
#             direct = p2 in adj[p1]
#             # find bridge
#             bridge = None
#             if not direct:
#                 common = adj[p1] & adj[p2]
#                 if common:
#                     bridge = common
#             if direct:
#                 print(f"  D{stab[i]}({p1}) - D{stab[j]}({p2}): DIRECT ✅")
#             elif bridge:
#                 print(f"  D{stab[i]}({p1}) - D{stab[j]}({p2}): via bridge {bridge}")
#             else:
#                 # look for 2-hop bridges
#                 for mid in adj[p1]:
#                     for mid2 in adj[mid]:
#                         if mid2 in adj[p2] and mid2 != p1:
#                             print(f"  D{stab[i]}({p1}) - D{stab[j]}({p2}): 2-hop {p1}→{mid}→{mid2}→{p2}")
#                             break
#                     else:
#                         continue
#                     break
#                 else:
#                     print(f"  D{stab[i]}({p1}) - D{stab[j]}({p2}): NO PATH FOUND ❌")

# # Weight-2 stabilizers
# print("\n=== Weight-2 stabilizer connectivity ===")
# for typ, stab in all_stabs:
#     if len(stab) != 2:
#         continue
#     p1, p2 = D[stab[0]], D[stab[1]]
#     direct = p2 in adj[p1]
#     bridge = adj[p1] & adj[p2]
#     print(f"{typ} stab {stab} -> {p1},{p2}: direct={direct}, bridge={bridge}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit import transpile
# import sys
# sys.path.append('.')
# from ibm_experiment.circuits.qiskit_surface_code_generator import SurfaceCodeCircuit

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')

# sc = SurfaceCodeCircuit(distance=3, num_rounds=3)
# qc = sc.build_circuit(initial_state=0)

# # 방법 1: 기본 transpile (현재 방식)
# tqc1 = transpile(qc, backend=backend, optimization_level=3)
# ops1 = tqc1.count_ops()
# cx1 = ops1.get('cx',0) + ops1.get('ecr',0) + ops1.get('cz',0)
# print(f"Default layout opt=3: depth={tqc1.depth()}, 2q={cx1}")

# # 방법 2: data를 vertices에 배치하는 initial_layout
# # 논리 큐빗 0-8 = data, 9-16 = ancilla, ...
# # data를 vertices 22,24,26,41,43,45,60,62,64에 강제 배치
# # ancilla는 인접한 links에 배치

# # SurfaceCodeCircuit의 큐빗 순서를 먼저 확인
# print(f"\nCircuit registers:")
# for reg in qc.qregs:
#     print(f"  {reg.name}: {reg.size} qubits")

# # 회로의 큐빗 수
# n = qc.num_qubits
# print(f"Total qubits: {n}")

# # initial_layout: 논리 큐빗 → 물리 큐빗 매핑
# # data qubits = 0-8 (가정), ancilla = 나머지
# # 여러 layout 시도
# layouts = {
#     "center_vertices": {
#         # data on vertices, ancilla on nearby positions
#         0: 22, 1: 24, 2: 26,
#         3: 41, 4: 43, 5: 45,
#         6: 60, 7: 62, 8: 64,
#         # ancilla + extra on surrounding qubits
#         9: 23, 10: 25, 11: 34, 12: 44,
#         13: 42, 14: 53, 15: 54, 16: 61,
#         17: 63, 18: 15, 19: 16, 20: 40,
#         21: 46, 22: 59, 23: 65, 24: 21,
#         25: 27, 26: 33, 27: 38, 28: 72,
#         29: 39, 30: 58, 31: 66, 32: 35,
#     },
# }

# for name, layout in layouts.items():
#     # layout이 큐빗 수에 맞는지 확인
#     if len(layout) < n:
#         # 부족한 큐빗은 빈 위치에 자동 배치
#         used = set(layout.values())
#         free = [q for q in range(127) if q not in used]
#         for i in range(len(layout), n):
#             layout[i] = free.pop(0)
    
#     try:
#         tqc = transpile(qc, backend=backend, optimization_level=3, 
#                         initial_layout=layout)
#         ops = tqc.count_ops()
#         cx = ops.get('cx',0) + ops.get('ecr',0) + ops.get('cz',0)
#         print(f"\n{name}: depth={tqc.depth()}, 2q={cx}")
#     except Exception as e:
#         print(f"\n{name}: ERROR - {e}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit import transpile
# from qiskit.circuit.library import XGate
# import sys
# sys.path.append('.')
# from ibm_experiment.circuits.qiskit_surface_code_generator import SurfaceCodeCircuit

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')

# sc = SurfaceCodeCircuit(distance=3, num_rounds=3)
# qc = sc.build_circuit(initial_state=0)

# # Qiskit expects list: initial_layout[logical_idx] = physical_qubit
# # 33 logical qubits → 33 physical positions
# # data(0-8) on vertices, ancilla 3라운드분(9-32) on nearby

# layout_list = [
#     # data 0-8: vertices
#     22, 24, 26, 41, 43, 45, 60, 62, 64,
#     # anc_r0 (9-16): 8 ancilla for round 0
#     23, 25, 34, 44, 42, 53, 54, 61,
#     # anc_r1 (17-24): 8 ancilla for round 1
#     15, 16, 40, 46, 63, 65, 59, 72,
#     # anc_r2 (25-32): 8 ancilla for round 2
#     21, 27, 33, 38, 35, 39, 58, 66,
# ]

# print(f"Layout size: {len(layout_list)}, circuit qubits: {qc.num_qubits}")
# assert len(layout_list) == qc.num_qubits

# # Default (no layout)
# tqc0 = transpile(qc, backend=backend, optimization_level=3)
# ops0 = tqc0.count_ops()
# cx0 = sum(ops0.get(g, 0) for g in ['cx','ecr','cz'])
# print(f"Default opt=3:         depth={tqc0.depth()}, 2q_gates={cx0}")

# # With manual layout
# for opt in [1, 2, 3]:
#     tqc = transpile(qc, backend=backend, optimization_level=opt,
#                     initial_layout=layout_list)
#     ops = tqc.count_ops()
#     cx = sum(ops.get(g, 0) for g in ['cx','ecr','cz'])
#     print(f"Manual layout opt={opt}: depth={tqc.depth()}, 2q_gates={cx}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit import transpile
# import sys
# sys.path.append('.')
# from ibm_experiment.circuits.qiskit_surface_code_generator import SurfaceCodeCircuit

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')

# sc = SurfaceCodeCircuit(distance=3, num_rounds=3)
# qc = sc.build_circuit(initial_state=0)

# best_depth = float('inf')
# best_seed = -1
# best_cx = 0

# for seed in range(200):
#     tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=seed)
#     ops = tqc.count_ops()
#     cx = sum(ops.get(g, 0) for g in ['cx','ecr','cz'])
#     depth = tqc.depth()
#     if depth < best_depth:
#         best_depth = depth
#         best_seed = seed
#         best_cx = cx
#         print(f"  New best! seed={seed}: depth={depth}, 2q={cx}")

# print(f"\n=== Best: seed={best_seed}, depth={best_depth}, 2q={best_cx} ===")
# print(f"=== vs paper fold-unfold: depth ~42 ===")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# degrees = {q: len(adj[q]) for q in adj}

# # 논문 Fig 1(b)의 unit cell:
# # A(data) - Bridge - B(ancilla_Z) - C(bridge) - D(ancilla_X) - E(bridge) - F(data)
# # 위아래로 data가 연결됨

# # 우리 배치에서 unit cell을 찾기:
# # vertex(deg-3) - link(deg-2) - vertex(deg-3) 체인

# # Row 1-2 사이 수직 연결 확인
# print("=== Vertical chains (Row1 → Row2 via links) ===")
# row1 = [20, 22, 24, 26, 28, 30]
# row2 = [39, 41, 43, 45, 47, 49]

# for v1 in row1:
#     for link in adj[v1]:
#         if degrees[link] == 2:
#             for v2 in adj[link]:
#                 if v2 != v1 and v2 in row2:
#                     print(f"  {v1} --[{link}]-- {v2}")

# print("\n=== Full heavy-hex chain: Row0-Row1-Row2-Row3 ===")
# row0 = [4, 8, 12]
# row3 = [58, 60, 62, 64, 66, 68]

# # Row0 → Row1
# for v0 in row0:
#     for link in adj[v0]:
#         if degrees[link] == 2:
#             for v1 in adj[link]:
#                 if v1 != v0 and v1 in row1:
#                     # v1 → Row2
#                     for link2 in adj[v1]:
#                         if degrees[link2] == 2 and link2 != link:
#                             for v2 in adj[link2]:
#                                 if v2 != v1 and v2 in row2:
#                                     print(f"  {v0}--[{link}]--{v1}  horizontal...  then {v1}--???--{v2}")

# # 전체 체인 하나 출력: 수직선 하나
# print("\n=== Example vertical chain ===")
# def trace_vertical(start, adj, degrees):
#     """수직으로 연결된 체인을 추적"""
#     chain = [start]
#     visited = {start}
#     current = start
#     # 아래로 내려가기
#     while True:
#         found = False
#         for n in adj[current]:
#             if n not in visited:
#                 # 수직 link를 찾기 (horizontal이 아닌 것)
#                 if degrees[n] == 2:
#                     for nn in adj[n]:
#                         if nn not in visited and degrees[nn] == 3:
#                             # horizontal이 아닌지 확인
#                             # (같은 row가 아닌지)
#                             chain.extend([n, nn])
#                             visited.add(n)
#                             visited.add(nn)
#                             current = nn
#                             found = True
#                             break
#             if found:
#                 break
#         if not found:
#             break
#     return chain

# # 15는 4↔22를 잇는 link
# chain = [4, 15, 22, 23, 24, 34, 43, 44, 45, 54, 64, 63, 62, 72]
# print("  Manual chain: " + " - ".join(str(q) for q in chain))
# for i in range(len(chain)-1):
#     ok = chain[i+1] in adj[chain[i]]
#     print(f"    {chain[i]}→{chain[i+1]}: {'✅' if ok else '❌'}")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# # 새 배치: data on specific vertices, ancilla on others
# data = {
#     0: 20, 1: 24, 2: 28,
#     3: 39, 4: 43, 5: 47,
#     6: 58, 7: 62, 8: 66,
# }

# # Ancilla: bulk (weight-4 stabilizers)
# # Z ancilla between horizontal data pairs
# # X ancilla between horizontal data pairs (alternating)
# z_ancilla_bulk = {
#     0: 22,  # between D0(20) and D1(24)
#     1: 45,  # between D4(43) and D5(47)
#     2: 60,  # between D6(58) and D7(62)
# }
# x_ancilla_bulk = {
#     0: 41,  # between D3(39) and D4(43)
#     1: 26,  # between D1(24) and D2(28)
#     2: 64,  # between D7(62) and D8(66)
# }

# # Boundary ancilla needed for weight-2 stabilizers
# # Z[0,1] boundary, Z[7,8] boundary, X[3,6] boundary, X[2,5] boundary

# print("=== Bulk Z stabilizers ===")
# # Z stabilizer = Z_left Z_right (after fold from weight-4)
# # Z stab 0: D0,D1,D3,D4 → fold to D1,D3 → ancilla 22 measures
# # Check: can 22 reach D1(24) and D3(39)?

# print("Z_anc 22: neighbors =", sorted(adj[22]))
# print("Z_anc 45: neighbors =", sorted(adj[45]))
# print("Z_anc 60: neighbors =", sorted(adj[60]))

# print("\n=== Bulk X stabilizers ===")
# print("X_anc 41: neighbors =", sorted(adj[41]))
# print("X_anc 26: neighbors =", sorted(adj[26]))
# print("X_anc 64: neighbors =", sorted(adj[64]))

# print("\n=== Bridge qubits and connectivity ===")
# bridges = [21, 23, 25, 27, 33, 34, 35, 40, 42, 44, 46, 53, 54, 59, 61, 63, 65]
# for b in bridges:
#     ns = sorted(adj[b])
#     print(f"  Bridge {b}: neighbors={ns}")

# print("\n=== Data-to-Data direct connectivity (via bridges) ===")
# for i in range(9):
#     for j in range(i+1, 9):
#         p1, p2 = data[i], data[j]
#         shared = adj[p1] & adj[p2]
#         if shared:
#             print(f"  D{i}({p1}) - D{j}({p2}): bridge {shared}")

# print("\n=== Data-to-Ancilla connectivity ===")
# for label, anc_dict in [("Z", z_ancilla_bulk), ("X", x_ancilla_bulk)]:
#     for idx, anc in anc_dict.items():
#         for i in range(9):
#             p = data[i]
#             if p in adj[anc]:
#                 print(f"  {label}_anc{idx}({anc}) - D{i}({p}): DIRECT")
#             else:
#                 shared = adj[anc] & adj[p]
#                 if shared:
#                     print(f"  {label}_anc{idx}({anc}) - D{i}({p}): via {shared}")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# # Verify all needed CX pairs for the fold-unfold scheme
# needed_cx = {
#     # Weight-2 boundary Z
#     "Z2: D0→bridge21":  (20, 21),
#     "Z2: bridge21→anc22": (21, 22),
#     "Z2: D1→bridge23":  (24, 23),
#     "Z2: bridge23→anc22": (23, 22),
    
#     # Weight-2 boundary Z  
#     "Z3: D7→bridge63":  (62, 63),
#     "Z3: bridge63→anc64": (63, 64),
#     "Z3: D8→bridge65":  (66, 65),
#     "Z3: bridge65→anc64": (65, 64),
    
#     # Weight-2 boundary X
#     "X2: D3→bridge40":  (39, 40),
#     "X2: bridge40→anc41": (40, 41),
#     "X2: D6→bridge53":  (58, 53),  # D6=58, anc41 via 53
#     "X2: bridge53→anc41": (53, 41),  # wait, 53 connects 41 and 60
    
#     "X3: D2→bridge27":  (28, 27),
#     "X3: bridge27→anc26": (27, 26),
#     "X3: D5→bridge46":  (47, 46),
#     "X3: bridge46→anc45": (46, 45),  # wrong ancilla?
    
#     # Fold CXs for weight-4
#     "Fold D0→D3 via bridge33 (1)": (20, 33),
#     "Fold D0→D3 via bridge33 (2)": (33, 39),
#     "Fold D1→D4 via bridge34 (1)": (24, 34),
#     "Fold D1→D4 via bridge34 (2)": (34, 43),
#     "Fold D2→D5 via bridge35 (1)": (28, 35),
#     "Fold D2→D5 via bridge35 (2)": (35, 47),
    
#     # After fold, measure weight-2:
#     # Z0 folded: Z_D3 Z_D4, measure with anc between D3,D4
#     "Z0 meas: D3→bridge40": (39, 40),
#     "Z0 meas: bridge40→anc41": (40, 41),
#     "Z0 meas: D4→bridge42": (43, 42),
#     "Z0 meas: bridge42→anc41": (42, 41),
    
#     # Z1 folded: Z_D6 Z_D7 (fold D3→D6, D4→D7)
#     "Z1 meas: D6→bridge59": (58, 59),
#     "Z1 meas: bridge59→anc60": (59, 60),
#     "Z1 meas: D7→bridge61": (62, 61),
#     "Z1 meas: bridge61→anc60": (61, 60),
    
#     # X0 folded: X_D1 X_D4 (fold D0→D1, D3→D4)
#     # anc between D1,D4 = bridge34 (can we use link as ancilla?)
#     "X0 meas: D1→bridge34": (24, 34),
#     "X0 meas: bridge34←D4": (43, 34),
#     # Or use ancilla 22? No, wrong type.
    
#     # X1 folded: X_D5 X_D8 (fold D4→D5, D7→D8)
#     "X1 meas: D5→bridge54": (47, 54),  # wait, 54 connects 45 and 64
#     "X1 meas: D8→bridge65": (66, 65),
# }

# print("=== CX Connectivity Verification ===")
# all_ok = True
# for label, (q1, q2) in sorted(needed_cx.items()):
#     ok = q2 in adj[q1]
#     if not ok:
#         all_ok = False
#     print(f"  {'✅' if ok else '❌'} {label}: CX({q1}→{q2})")

# print(f"\nAll OK: {all_ok}")

# # Also check: 58→53 (D6 to bridge53)
# print(f"\n58 neighbors: {sorted(adj[58])}")
# print(f"53 neighbors: {sorted(adj[53])}")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# # 문제 1: X2 boundary X(D3,D6) = X(39,58)
# # anc41은 D3에만 도달, D6에 못 감
# # anc60은 D6에만 도달, D3에 못 감
# # → fold 방식: CX(D6→D3) via 58→59→60→53→41→40→39 (너무 긺)
# # → 대안: 중간 ancilla로 53(link)을 사용?

# print("=== Option: use bridge 53 as measurement point for X2 ===")
# print(f"  53 neighbors: {sorted(adj[53])}")  # [41, 60]
# # 53은 41과 60에만 연결. D3(39)나 D6(58)에 직접 도달 불가

# # 대안: stabilizer 재할당
# # X2(D3,D6) → ancilla를 41이 아닌 다른 곳에?
# # D3=39: neighbors = {33, 40}
# # D6=58: neighbors = {57, 59, 71}
# # 공통 도달 가능한 ancilla? 없음

# # 근본 해결: 논문처럼 fold 방향을 바꿔야 함
# # fold은 data→data CX인데, D6→D3은 거리 문제

# # 논문의 접근: half-round에서 절반만 측정
# # Round 1/2: Z stabilizers (Z0, Z1, boundary Z2, Z3)
# # Round 2/2: X stabilizers (X0, X1, boundary X2, X3)
# # 이때 fold-unfold의 CNOT이 겹치면서 상쇄

# # 실제로 필요한 건: 각 weight-4의 fold CX가 가능한지
# # 현재 fold CX (vertical):
# print("\n=== Vertical fold CXs ===")
# folds = [
#     ("D0→D3", 20, 33, 39),  # bridge 33
#     ("D1→D4", 24, 34, 43),  # bridge 34
#     ("D2→D5", 28, 35, 47),  # bridge 35
# ]
# for label, d_top, bridge, d_bot in folds:
#     ok1 = bridge in adj[d_top]
#     ok2 = d_bot in adj[bridge]
#     print(f"  {label}: {d_top}→{bridge}({'✅' if ok1 else '❌'}) → {d_bot}({'✅' if ok2 else '❌'})")

# # Row 2→3 vertical connections
# print("\n=== Row2→Row3 vertical connections ===")
# row2_data = [39, 43, 47]
# row3_data = [58, 62, 66]
# for d2 in row2_data:
#     for n in sorted(adj[d2]):
#         for nn in sorted(adj[n]):
#             if nn in row3_data:
#                 print(f"  D({d2}) → {n} → D({nn})")

# # 핵심: Row2→Row3은 vertex→link→vertex 경로가 staggered
# # 41→53→60, 45→54→64
# # 즉 D3(39)→40→41→53→60(anc)→... 으로 D6 방향
# # D5(47)→46→45→54→64(anc)→... 으로 D8 방향

# print("\n=== Alternative: 2-stage fold ===")
# # Weight-4 Z1(D3,D4,D6,D7):
# # Stage 1: fold D3→D6 via 39→40→41→53→60 (but 60 is ancilla, not data)
# # 이건 안됨. D6=58인데 60은 ancilla

# # 진짜 문제: Row2 data와 Row3 data 사이에 직접 bridge가 없음
# # Row1→2: bridge 33,34,35 (직접)
# # Row2→3: 41→53→60(anc), 45→54→64(anc) 
# #          data(39)→bridge(40)→vertex(41)→bridge(53)→vertex(60) ← 60은 data가 아닌 ancilla!

# print("\n=== Key insight: vertical data-data connections ===")
# print("Row1→Row2: D0→D3(bridge33), D1→D4(bridge34), D2→D5(bridge35)")
# print("Row2→Row3: NO direct data-data bridge!")
# print("  D3(39)→40→41(anc)→53→60(anc) → not D6(58)")
# print("  D4(43)→44→45(anc)→54→64(anc) → not D7(62)")
# print("  D5(47)→46→45(anc)→... → not D8(66)")
# print()
# print("Row2 data to Row3 data: must go through 2 vertices (both ancilla)")
# print("This means fold-unfold for Row2→Row3 stabilizers is NOT possible")
# print("with a single bridge hop!")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# # 새 배치: 연속 vertex
# data = {0:22, 1:24, 2:26, 3:41, 4:43, 5:45, 6:60, 7:62, 8:64}

# # Surface code d=3 stabilizers
# x_stabs = [[0,1,3,4], [4,5,7,8], [3,6], [2,5]]
# z_stabs = [[1,2,4,5], [3,4,6,7], [0,1], [7,8]]

# # === Step 1: 수평 fold CX 검증 ===
# # fold은 같은 row에서 인접 data pair 사이
# fold_pairs = [
#     ("D0↔D1", 0, 1), ("D1↔D2", 1, 2),
#     ("D3↔D4", 3, 4), ("D4↔D5", 4, 5),
#     ("D6↔D7", 6, 7), ("D7↔D8", 7, 8),
# ]

# print("=== Horizontal fold CX (data↔data via bridge) ===")
# for label, i, j in fold_pairs:
#     p1, p2 = data[i], data[j]
#     shared = adj[p1] & adj[p2]
#     print(f"  {label}: D{i}({p1})↔D{j}({p2}) bridge={shared} {'✅' if shared else '❌'}")

# # === Step 2: Fold 후 측정에 필요한 vertical 연결 ===
# # X0[0,1,3,4]: fold D0→D1, D3→D4 → measure D1(24), D4(43)
# # X1[4,5,7,8]: fold D4→D5, D7→D8 → measure D5(45), D8(64)
# # Z0[1,2,4,5]: fold D1→D2, D4→D5 → measure D2(26), D5(45)
# # Z1[3,4,6,7]: fold D3→D4, D6→D7 → measure D4(43), D7(62)

# print("\n=== Vertical measurement after fold ===")
# meas_pairs = [
#     ("X0: D1,D4", 1, 4),
#     ("X1: D5,D8", 5, 8),
#     ("Z0: D2,D5", 2, 5),
#     ("Z1: D4,D7", 4, 7),
# ]
# for label, i, j in meas_pairs:
#     p1, p2 = data[i], data[j]
#     shared = adj[p1] & adj[p2]
#     if shared:
#         print(f"  {label}: D{i}({p1}),D{j}({p2}) shared_bridge={shared} ✅")
#     else:
#         # 2-hop search: p1→mid→p2
#         for mid in adj[p1]:
#             if mid in adj[p2]:
#                 pass  # already checked
#             for mid2 in adj[mid]:
#                 if mid2 in adj[p2] and mid2 != p1:
#                     print(f"  {label}: D{i}({p1}),D{j}({p2}) 2-hop: {p1}→{mid}→{mid2}→{p2}")
#                     break
#             else:
#                 continue
#             break
#         else:
#             print(f"  {label}: D{i}({p1}),D{j}({p2}) NO PATH ❌")

# # === Step 3: Boundary stabilizer 연결 ===
# print("\n=== Boundary stabilizers ===")
# boundary = [
#     ("X2[3,6]", 3, 6),
#     ("X3[2,5]", 2, 5),
#     ("Z2[0,1]", 0, 1),
#     ("Z3[7,8]", 7, 8),
# ]
# for label, i, j in boundary:
#     p1, p2 = data[i], data[j]
#     shared = adj[p1] & adj[p2]
#     print(f"  {label}: D{i}({p1}),D{j}({p2}) bridge={shared} {'✅' if shared else '❌'}")

# # === Step 4: 가능한 ancilla 위치 ===
# print("\n=== Available ancilla positions ===")
# all_data = set(data.values())
# for label, i, j in meas_pairs + boundary:
#     p1, p2 = data[i], data[j]
#     # 두 data에 모두 도달 가능한 큐빗
#     reachable_from_p1 = adj[p1]
#     reachable_from_p2 = adj[p2]
#     # 2-hop 도달
#     reach2_p1 = set()
#     for n in adj[p1]:
#         reach2_p1.update(adj[n])
#     reach2_p2 = set()
#     for n in adj[p2]:
#         reach2_p2.update(adj[n])
    
#     common_1hop = reachable_from_p1 & reachable_from_p2
#     common_2hop = reach2_p1 & reach2_p2 - all_data
#     print(f"  {label}: 1-hop common={common_1hop - all_data}, 2-hop common (non-data)={common_2hop - all_data - common_1hop}")



# from collections import defaultdict
# from qiskit_ibm_runtime import QiskitRuntimeService

# service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
# backend = service.backend('ibm_yonsei')
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)

# data_set = {22,24,26,41,43,45,60,62,64}

# # X3 relay path: 26→27→28→35→47→46→45
# path = [26, 27, 28, 35, 47, 46, 45]
# print("=== X3 relay path verification ===")
# for i in range(len(path)-1):
#     q1, q2 = path[i], path[i+1]
#     ok = q2 in adj[q1]
#     is_data = "DATA" if q2 in data_set else "non-data"
#     print(f"  {q1} → {q2}: {'✅' if ok else '❌'} ({is_data})")

# # 중간 큐빗들이 data가 아닌지 확인
# relay_qubits = [27, 28, 35, 47, 46]
# print(f"\nRelay qubits: {relay_qubits}")
# print(f"Any overlap with data? {set(relay_qubits) & data_set}")

# # 전체 CX 수 추정
# print("\n=== Estimated CX per cycle ===")
# print("Weight-4 fold-unfold (×4):")
# print("  Each: 2×2 fold CX + 2 measure CX + 2×2 unfold CX = 12 CX")
# print("  Total: 48 CX")
# print("Weight-2 bridge (×3):")
# print("  Each: 2 CX via bridge + 2 CX unfold = 4 CX")
# print("  Total: 12 CX")
# print("Weight-2 X3 relay:")
# print("  SWAP relay (5 hops × 3 CX) + 2 CX measure = 17 CX")
# print("  Or: just extend fold from weight-4")
# print(f"\nEstimated total per cycle: ~77 CX")
# print(f"Naive transpile: ~126 CX/cycle (378/3)")
# print(f"Improvement: ~40%")

# # 더 나은 접근: X3를 fold 방식으로 처리
# # X3[2,5]를 weight-4의 일부로 흡수는 안 됨
# # 하지만 X1[4,5,7,8]의 unfold 직후에 D5의 X 정보가 여전히 available
# # → software decoding에서 추론 가능?
# print("\n=== Alternative: skip X3, use software decoding ===")
# print("If X3 is not measured, code becomes (3,2) instead of (3,3)")
# print("This reduces Z-error correction capability")
# print("NOT recommended")



    # from qiskit_ibm_runtime import QiskitRuntimeService
    # from qiskit import transpile
    # import sys
    # sys.path.insert(0, 'ibm_experiment/circuits')
    # from heavyhex_surface_code import HeavyHexSurfaceCode

    # service = QiskitRuntimeService(instance="yonsei_internal-dedicated")
    # backend = service.backend('ibm_yonsei')

    # sc = HeavyHexSurfaceCode(distance=3, num_cycles=3)
    # qc = sc.build_circuit(initial_state=0)

    # print(f"Pre-transpile: depth={qc.depth()}, CX={qc.count_ops().get('cx',0)}")

    # for opt in [0, 1, 2, 3]:
    #     tqc = transpile(qc, backend=backend, optimization_level=opt)
    #     ops = tqc.count_ops()
    #     two_q = ops.get('ecr',0) + ops.get('cz',0) + ops.get('cx',0)
    #     swap = ops.get('swap', 0)
    #     print(f"  opt={opt}: depth={tqc.depth()}, 2q_gates={two_q}, swaps={swap}")

    # # Naive 비교 (기존 회로)
    # sys.path.insert(0, 'ibm_experiment')
    # from qiskit_surface_code_generator import SurfaceCodeCircuit
    # sc_naive = SurfaceCodeCircuit(distance=3, num_rounds=3)
    # qc_naive = sc_naive.build_circuit(initial_state=0)
    # tqc_naive = transpile(qc_naive, backend=backend, optimization_level=3)
    # ops_naive = tqc_naive.count_ops()
    # two_q_naive = ops_naive.get('ecr',0) + ops_naive.get('cz',0) + ops_naive.get('cx',0)
    # print(f"\n  Naive opt=3: depth={tqc_naive.depth()}, 2q_gates={two_q_naive}")
    # print(f"\n  핵심: 2q_gates {two_q_naive} → 216 = {(two_q_naive-216)/two_q_naive*100:.0f}% 감소")



# from qiskit import transpile, QuantumCircuit
# from qiskit_aer import AerSimulator
# import sys
# sys.path.insert(0, 'ibm_experiment/circuits')
# from heavyhex_surface_code import HeavyHexSurfaceCode

# sc = HeavyHexSurfaceCode(distance=3, num_cycles=3)
# qc = sc.build_circuit(initial_state=0)

# # AerSimulator에 큐빗 수 지정
# sim = AerSimulator(n_qubits=65)
# result = sim.run(qc, shots=1000).result()
# counts = result.get_counts()

# total = sum(counts.values())
# correct = 0
# for bitstr, count in counts.items():
#     parts = bitstr.split()
#     if len(parts) >= 2:
#         data_bits = parts[0]
#         if all(b == '0' for b in data_bits):
#             correct += count
#     else:
#         data_bits = bitstr[-9:]
#         if all(b == '0' for b in data_bits):
#             correct += count

# print(f"Total shots: {total}")
# print(f"Correct |0>_L: {correct}/{total} = {correct/total*100:.1f}%")
# print(f"\nTop 5 outcomes:")
# for bitstr, cnt in sorted(counts.items(), key=lambda x: -x[1])[:5]:
#     print(f"  {bitstr}: {cnt}")



from qiskit_aer import AerSimulator
import sys
sys.path.insert(0, 'ibm_experiment/circuits')
from heavyhex_surface_code import HeavyHexSurfaceCode

sc = HeavyHexSurfaceCode(distance=3, num_cycles=1)  # 1 cycle만
qc = sc.build_circuit(initial_state=0)

sim = AerSimulator(n_qubits=65)
result = sim.run(qc, shots=1000).result()
counts = result.get_counts()

# Logical Z = parity of D0(22), D3(41), D6(60)
# data register 순서: D0=idx0, D3=idx3, D6=idx6
logical_z_correct = 0
total = sum(counts.values())

for bitstr, count in counts.items():
    parts = bitstr.split()
    data_bits = parts[0] if len(parts) >= 2 else bitstr[-9:]
    
    # data_bits는 MSB-first, idx 8...0
    # D0=idx0 → data_bits[-1], D3=idx3 → data_bits[-4], D6=idx6 → data_bits[-7]
    d0 = int(data_bits[-1])
    d3 = int(data_bits[-4])
    d6 = int(data_bits[-7])
    logical_z = (d0 + d3 + d6) % 2
    
    if logical_z == 0:
        logical_z_correct += count

print(f"Total: {total}")
print(f"Logical Z = 0 (correct): {logical_z_correct}/{total} = {logical_z_correct/total*100:.1f}%")

# 또한 syndrome 값 확인
print(f"\nTop 10 outcomes:")
for bitstr, cnt in sorted(counts.items(), key=lambda x: -x[1])[:10]:
    parts = bitstr.split()
    data_bits = parts[0] if len(parts) >= 2 else bitstr[-9:]
    syn_bits = parts[1] if len(parts) >= 2 else bitstr[:-9]
    d0 = int(data_bits[-1])
    d3 = int(data_bits[-4])
    d6 = int(data_bits[-7])
    lz = (d0+d3+d6)%2
    print(f"  data={data_bits} syn={syn_bits} logical_Z={lz} count={cnt}")


sim = AerSimulator(n_qubits=65)

for cycles in [1, 3, 5]:
    sc = HeavyHexSurfaceCode(distance=3, num_cycles=cycles)
    qc = sc.build_circuit(initial_state=0)
    result = sim.run(qc, shots=1000).result()
    counts = result.get_counts()
    
    correct = 0
    total = sum(counts.values())
    for bitstr, count in counts.items():
        parts = bitstr.split()
        data_bits = parts[0] if len(parts) >= 2 else bitstr[-9:]
        d0 = int(data_bits[-1])
        d3 = int(data_bits[-4])
        d6 = int(data_bits[-7])
        if (d0 + d3 + d6) % 2 == 0:
            correct += count
    
    depth = qc.depth()
    cx = qc.count_ops().get('cx', 0)
    print(f"Cycles={cycles}: Logical_Z=0 {correct}/{total}={correct/total*100:.0f}%, depth={depth}, CX={cx}")