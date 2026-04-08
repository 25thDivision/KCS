# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_boston")
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)
# degrees = {q: len(adj[q]) for q in adj}

# # 논문 Fig 1(a) 구조:
# # Heavy-hex unit cell (수평):
# #   A(data) - Bridge - B(ancilla) - C(bridge) - D(ancilla) - E(bridge) - F(data)
# #
# # Vertex 역할: data와 ancilla가 번갈아 배치
# # Link 역할: bridge (측정 안 함)
# #
# # (3,3) surface code:
# #   9 data + 8 ancilla = 17 qubits on vertices
# #   나머지 = bridge on links

# # ibm_boston 수평 체인 (Row C): v41-v43-v45-v47-v49-v51-v53
# # 논문 방식: v41=data, v43=ancilla, v45=data, v47=ancilla, v49=data, v51=ancilla, v53=data
# # 또는:     v41=ancilla, v43=data, v45=ancilla, v47=data, ...

# # 전체 vertex 행 출력
# rows_v = {
#     "A": [3, 7, 11],
#     "B": [21, 23, 25, 27, 29, 31, 33],
#     "C": [41, 43, 45, 47, 49, 51, 53],
#     "D": [61, 63, 65, 67, 69, 71, 73],
#     "E": [81, 83, 85, 87, 89, 91, 93],
#     "F": [101, 103, 105, 107, 109, 111, 113],
#     "G": [121, 123, 125, 127, 129, 131, 133],
#     "H": [143, 147, 151],
# }

# # 대각선 연결 (row B → row C, row C → row D)
# print("=== Diagonal connections ===")
# for label, (r_up, r_down) in [("B→C", ("B","C")), ("C→D", ("C","D")), ("D→E", ("D","E"))]:
#     print(f"\n{label}:")
#     for vu in rows_v[r_up]:
#         for link in adj[vu]:
#             if degrees[link] == 2:
#                 for vd in adj[link]:
#                     if vd in rows_v[r_down]:
#                         print(f"  v{vu} --[{link}]-- v{vd}")

# # 논문의 (3,3) code는 3행 5열 영역 사용
# # Fig 1(a): yellow boundary = (3,3)
# # 3행: Row B, C, D 중 3행
# # 5열: 5개 vertex per row

# # 논문 방식 role assignment (alternating):
# # Row B: anc, DATA, anc, DATA, anc, DATA, anc  (7 vertices)
# # Row C: DATA, anc, DATA, anc, DATA, anc, DATA (7 vertices)  
# # Row D: anc, DATA, anc, DATA, anc, DATA, anc  (7 vertices)

# # (3,3) sublattice: 3행 × 5열 = 15 vertices
# # Row B[1:6]: v23, v25, v27, v29, v31  (5개)
# # Row C[0:5]: v41, v43, v45, v47, v49  (5개)  
# # Row D[1:6]: v63, v65, v67, v69, v71  (5개)

# # Role assignment for (3,3):
# # Row B: D23=data, D25=anc_Z, D27=data, D29=anc_Z, D31=data
# # Row C: D41=anc_X, D43=data, D45=anc_X, D47=data, D49=anc_X
# # Row D: D63=data, D65=anc_Z, D67=data, D69=anc_Z, D71=data

# # 이러면 data = 23,27,31,43,47,63,67,71 = 8개... 9개가 안 됨
# # 논문에서는 boundary에 추가 data가 있을 수 있음

# # 아니면 다른 패턴:
# # Row B: anc, data, anc, data, anc
# # Row C: data, anc, data, anc, data
# # Row D: anc, data, anc, data, anc

# print("\n=== Paper-style alternating assignment ===")
# print("Option 1: Row B starts with ancilla")
# for label, row, start_data in [
#     ("B", [23, 25, 27, 29, 31], False),
#     ("C", [41, 43, 45, 47, 49], True),
#     ("D", [63, 65, 67, 69, 71], False),
# ]:
#     roles = []
#     for i, v in enumerate(row):
#         is_data = (i % 2 == 0) == start_data
#         roles.append(("DATA" if is_data else "ANC", v))
#     data_count = sum(1 for r, _ in roles if r == "DATA")
#     anc_count = sum(1 for r, _ in roles if r == "ANC")
#     print(f"  Row {label}: {roles}  (data={data_count}, anc={anc_count})")

# data_all = [v for label, row, start_data in [("B", [23,25,27,29,31], False), ("C", [41,43,45,47,49], True), ("D", [63,65,67,69,71], False)] for i, v in enumerate(row) if (i%2==0) == start_data]
# anc_all = [v for label, row, start_data in [("B", [23,25,27,29,31], False), ("C", [41,43,45,47,49], True), ("D", [63,65,67,69,71], False)] for i, v in enumerate(row) if (i%2==0) != start_data]
# print(f"\n  Total data: {len(data_all)} -> {data_all}")
# print(f"  Total anc:  {len(anc_all)} -> {anc_all}")

# # 대각선 연결로 어떤 data-ancilla 쌍이 직접 연결되는지
# print(f"\n=== Data-Ancilla direct diagonal connections ===")
# for d in data_all:
#     for link in adj[d]:
#         if degrees[link] == 2:
#             for a in adj[link]:
#                 if a in anc_all:
#                     print(f"  data({d}) --[{link}]-- anc({a})")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_boston")
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)
# degrees = {q: len(adj[q]) for q in adj}

# # 논문 방식 배치
# # Data: 7 vertices + 2 diagonal links = 9
# data_v = [25, 29, 41, 45, 49, 65, 69]   # vertex data
# data_l = [37, 38]                         # link data (diagonals)
# data_all = data_v + data_l                # 9 total

# # Ancilla: 8 vertices (전용, no sharing)
# anc_all = [23, 27, 31, 43, 47, 63, 67, 71]

# # Bridge: remaining links (fold-unfold용)
# bridge = [24, 26, 28, 30, 42, 44, 46, 48, 56, 57, 64, 66, 68, 70]

# print("=== Paper-style Layout ===")
# print(f"Data ({len(data_all)}):    {data_all}")
# print(f"Ancilla ({len(anc_all)}): {anc_all}")

# # 3x3 grid mapping
# # Rotated surface code d=3:
# #   D0  D1  D2
# #   D3  D4  D5
# #   D6  D7  D8
# #
# # Row B: D0=25, D1=29
# # Diagonal: D3=37 (link 25↔45), D4=38 (link 29↔49)  
# # Row C: D2=41, D5=45, (skip anc) D6=49
# # Row D: D7=65, D8=69
# # Wait, that's only 9 but the spatial arrangement needs to make sense...

# # Let me try a different mapping based on spatial positions:
# # Row top:    25 --- 29              (2 data)
# # Row mid:  41 - 37 - 45 - 38 - 49  (5 data, but 37/38 are diagonal links)
# # Row bot:    65 --- 69              (2 data)
# # Total: 9 data ✅

# # Map to standard d=3 grid:
# D = {
#     0: 25, 1: 37, 2: 41,   # top-left to bottom-left diagonal
#     3: 29, 4: 45, 5: 65,   # ?
#     6: 38, 7: 49, 8: 69,   # ?
# }
# # Hmm, spatial mapping needs care. Let me just check connectivity.

# # 실제 좌표를 기반으로 정렬
# # 각 qubit의 위치를 row/col로 추정
# print("\n=== Spatial positions ===")
# for q in sorted(data_all + anc_all):
#     role = "DATA" if q in data_all else "ANC"
#     ns = sorted(adj[q])
#     print(f"  q{q:3d} [{role:4s}]: neighbors={ns}, deg={degrees[q]}")

# # 핵심: 각 ancilla가 어떤 data에 도달 가능한지
# print("\n=== Ancilla → reachable Data (1-hop and via bridge) ===")
# for a in anc_all:
#     direct = [d for d in data_all if d in adj[a]]
#     via_bridge = []
#     for b in adj[a]:
#         if b not in data_all and b not in anc_all:  # bridge
#             for d in adj[b]:
#                 if d in data_all and d not in direct:
#                     via_bridge.append((d, b))
#     print(f"  anc {a}: direct={direct}, via_bridge={via_bridge}")

# # 각 data가 어떤 data에 도달 가능한지 (stabilizer pair)
# print("\n=== Data → Data connections (for stabilizer grouping) ===")
# for i, d1 in enumerate(data_all):
#     for d2 in data_all:
#         if d1 >= d2:
#             continue
#         # direct
#         if d2 in adj[d1]:
#             print(f"  D({d1})-D({d2}): DIRECT")
#         else:
#             # via bridge
#             common = set()
#             for b in adj[d1]:
#                 if b in adj[d2] and b not in data_all and b not in anc_all:
#                     common.add(b)
#             if common:
#                 print(f"  D({d1})-D({d2}): via bridge {common}")



from qiskit_ibm_runtime import QiskitRuntimeService
from collections import defaultdict

service = QiskitRuntimeService(instance="Yonsei_internal")
backend = service.backend("ibm_boston")
edges = backend.coupling_map.get_edges()

adj = defaultdict(set)
for q1, q2 in edges:
    adj[q1].add(q2)
    adj[q2].add(q1)
degrees = {q: len(adj[q]) for q in adj}

data_all = [25, 29, 37, 38, 41, 45, 49, 65, 69]
anc_all = [23, 27, 31, 43, 47, 63, 67, 71]

# 하단 연결 문제: 65와 69가 중단 data에 어떻게 도달하는가?
# 65 neighbors: [64, 66, 77]
# 69 neighbors: [68, 70, 78]
# 64 → anc63, 66 → anc67
# 이건 anc에 연결되지 data에 연결 안 됨

# 논문에서는 어떻게 했을까?
# 논문의 diagonal links: C→D
# v43 --[56]-- v63 (anc→anc)
# v47 --[57]-- v67 (anc→anc)
# v51 --[58]-- v71 (anc→anc... v51은 우리 layout에 없음)

# 가능한 해결: diagonal link 56, 57을 data로 사용
# 하지만 56은 anc43과 anc63 사이... 둘 다 anc이면 data로 전환?

# 다른 접근: 논문의 (3,3) sublattice가 정확히 어디인지 확인
# Fig 1(a)의 yellow boundary

# Row B-C 간 diagonal link를 data로 쓴 것처럼
# Row C-D 간 diagonal link도 data로 써야 하나?

# C→D diagonal links
print("=== C→D diagonal links ===")
row_c_v = [41, 43, 45, 47, 49, 51, 53]
row_d_v = [61, 63, 65, 67, 69, 71, 73]
for vc in row_c_v:
    for n in adj[vc]:
        if degrees[n] == 2:
            for vd in adj[n]:
                if vd in row_d_v:
                    role_c = "DATA" if vc in data_all else "ANC" if vc in anc_all else "NONE"
                    role_d = "DATA" if vd in data_all else "ANC" if vd in anc_all else "NONE"
                    print(f"  {vc}({role_c}) --[{n}]-- {vd}({role_d})")

# 만약 link 56, 57도 data로 사용하면?
print("\n=== If links 56, 57 become data ===")
extra_data = [56, 57]
for d in extra_data:
    ns = sorted(adj[d])
    print(f"  q{d}: neighbors={ns}")
    reachable_data = [n for n in ns if n in data_all]
    reachable_anc = [n for n in ns if n in anc_all]
    print(f"    reaches data: {reachable_data}")
    print(f"    reaches anc: {reachable_anc}")

# 그러면 data = 11개, anc = 6개... 비율이 안 맞음
# d=3 surface code는 정확히 9 data + 8 anc

# 대안: anc63, anc67을 data로, link56, link57을 anc로?
print("\n=== Alternative: swap roles ===")
print("If 63=data, 67=data, 56=anc, 57=anc:")
alt_data = [25, 29, 37, 38, 41, 45, 49, 63, 67]  # 9 data
alt_anc = [23, 27, 31, 43, 47, 56, 57, 71]        # 8 anc... but 65,69 orphaned

# Yet another: 4-row layout
print("\n=== 4-row approach ===")
print("Row A: [3, 7, 11]")
print("Row B: [21, 23, 25, 27, 29, 31, 33]")  
print("Row C: [41, 43, 45, 47, 49, 51, 53]")
print("Row D: [61, 63, 65, 67, 69, 71, 73]")

# 논문의 Fig 1(a)를 보면 (3,3)는 2.5행 정도를 사용
# 실제로는 B-C 사이의 diagonal data(37,38)가 "중간 행"

# 핵심 질문: 65, 69를 쓰지 않고 다른 9개를 고를 수 있는가?
# 25, 29 (Row B) + 37, 38 (diagonal) + 41, 43, 45, 47, 49 (Row C) = 9!
print("\n=== 2-row + diagonal approach (NO Row D) ===")
alt2_data = [25, 29, 37, 38, 41, 45, 49]  # 7... need 2 more
print(f"Only 7 data from B+diagonal+C. Need 2 more.")
print(f"Options: add 43, 47 as data → but they're ancilla")
print(f"Or: add boundary links 36(21→41), 40(41 boundary)")

# 실제로 논문 방식을 자세히 보면:
# (3,3) code는 5×5 격자가 아니라, heavy-hex에 맞게 변형된 code
# data qubit 배치가 표준과 다를 수 있음

print("\n=== All qubits in B-C region with roles ===")
region = set()
for v in [23, 25, 27, 29, 31, 41, 43, 45, 47, 49]:
    region.add(v)
    for n in adj[v]:
        region.add(n)

for q in sorted(region):
    in_data = q in data_all
    in_anc = q in anc_all
    role = "DATA" if in_data else "ANC" if in_anc else "---"
    ns = sorted(adj[q])
    ns_roles = []
    for n in ns:
        if n in data_all: ns_roles.append(f"{n}(D)")
        elif n in anc_all: ns_roles.append(f"{n}(A)")
        else: ns_roles.append(f"{n}")
    print(f"  q{q:3d} [{role:4s}] deg={degrees[q]}: {', '.join(ns_roles)}")