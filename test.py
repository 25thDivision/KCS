# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict

# for instance in ["Yonsei_internal", "yonsei_internal-dedicated", "Yonsei_internal-eu", "Y_BS_candid-dedicated", "Y_BS_candid"]:
#     service = QiskitRuntimeService(instance=instance)

#     for name in ["ibm_boston", "ibm_kingston", "ibm_pittsburgh", "ibm_fez", "ibm_marrakesh", "ibm_maimi", "ibm_aachen", "ibm_brussels", "ibm_strasbourg"]:
#         try:
#             backend = service.backend(name)
#             edges = backend.coupling_map.get_edges()
#             adj = defaultdict(set)
#             for q1, q2 in edges:
#                 adj[q1].add(q2)
#                 adj[q2].add(q1)
            
#             # 우리 회로에 필요한 CX 쌍 검증
#             needed = [
#                 (22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,35),
#                 (35,47),(47,46),(46,45),(45,44),(44,43),(43,42),(42,41),
#                 (41,40),(40,39),(41,53),(53,60),(60,59),(59,58),
#                 (60,61),(61,62),(62,63),(63,64),(64,65),(65,66),
#                 (24,34),(34,43),(45,54),(54,64),
#             ]
#             ok = sum(1 for a,b in needed if b in adj[a])
#             total = len(needed)
#             print(f"{instance}:{name}: {backend.num_qubits}q, {ok}/{total} links OK {'✅' if ok==total else '❌'}")
#         except Exception as e:
#             print(f"{instance}:{name}: ERROR - {e}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict

# # 크레딧 있는 instance들
# for inst in ["Yonsei_internal", "Y_BS_candid-dedicated"]:
#     try:
#         service = QiskitRuntimeService(instance=inst)
#         backends = service.backends()
#         for b in backends:
#             if b.num_qubits >= 127:
#                 edges = b.coupling_map.get_edges()
#                 adj = defaultdict(set)
#                 for q1, q2 in edges:
#                     adj[q1].add(q2)
#                     adj[q2].add(q1)
#                 degrees = defaultdict(int)
#                 for q in adj:
#                     degrees[q] = len(adj[q])
#                 from collections import Counter
#                 deg_dist = Counter(degrees.values())
#                 print(f"{inst}: {b.name} {b.num_qubits}q, edges={len(edges)}, degrees={deg_dist}")
#     except Exception as e:
#         print(f"{inst}: ERROR - {e}")



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
# vertices = sorted([q for q, d in degrees.items() if d == 3])
# links = sorted([q for q, d in degrees.items() if d == 2])

# print(f"Vertices (deg-3): {len(vertices)}")
# print(f"Links (deg-2): {len(links)}")

# # Vertex-to-vertex connections
# print("\n=== Vertex grid ===")
# # Find rows by tracing horizontal vertex chains
# visited = set()
# rows = []
# for start in vertices:
#     if start in visited:
#         continue
#     # trace horizontal chain
#     row = [start]
#     visited.add(start)
#     # go right
#     current = start
#     while True:
#         found = False
#         for link in adj[current]:
#             if degrees[link] == 2:
#                 for v in adj[link]:
#                     if v != current and degrees[v] == 3 and v not in visited:
#                         # check if horizontal (not vertical)
#                         # horizontal = same-row link
#                         row.append(v)
#                         visited.add(v)
#                         current = v
#                         found = True
#                         break
#             if found:
#                 break
#         if not found:
#             break
#     rows.append(sorted(row))

# rows.sort(key=lambda r: r[0])
# for i, row in enumerate(rows):
#     print(f"  Row {i}: {row}")

# # 중심부 3x3 영역 찾기
# # Eagle과 동일한 패턴 확인
# mid_row = len(rows) // 2
# print(f"\n=== Center rows ({mid_row-1} to {mid_row+1}) ===")
# for i in range(max(0, mid_row-2), min(len(rows), mid_row+3)):
#     print(f"  Row {i}: {rows[i]}")
#     for v in rows[i]:
#         ns = sorted(adj[v])
#         print(f"    q{v}: neighbors={ns}")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict
# from itertools import combinations

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_boston")
# edges = backend.coupling_map.get_edges()

# adj = defaultdict(set)
# for q1, q2 in edges:
#     adj[q1].add(q2)
#     adj[q2].add(q1)
# degrees = {q: len(adj[q]) for q in adj}
# vertices = sorted([q for q, d in degrees.items() if d == 3])

# # Step 1: Find all vertex-to-vertex connections via single link
# v2v = {}  # (v1, v2) -> link
# for v in vertices:
#     for link in adj[v]:
#         if degrees[link] == 2:
#             for v2 in adj[link]:
#                 if v2 != v and degrees[v2] == 3:
#                     if v < v2:
#                         v2v[(v, v2)] = link

# print(f"=== {len(v2v)} vertex-vertex connections ===")

# # Step 2: Build vertex adjacency graph
# v_adj = defaultdict(set)
# for (v1, v2) in v2v:
#     v_adj[v1].add(v2)
#     v_adj[v2].add(v1)

# # Step 3: Find 3x3 grid candidates
# # Need: 3 rows of 3 vertices each
# # Row: 3 vertices connected horizontally (v0-v1-v2)
# # Vertical: staggered connections between rows

# # Find horizontal triplets
# h_triplets = []
# for v in vertices:
#     for v2 in v_adj[v]:
#         for v3 in v_adj[v2]:
#             if v3 != v and v < v3:
#                 h_triplets.append(tuple(sorted([v, v2, v3])))

# h_triplets = list(set(h_triplets))
# print(f"\nFound {len(h_triplets)} horizontal triplets")

# # Step 4: For each pair of triplets, check if they form 2 rows with vertical connections
# best_layout = None
# best_score = -1

# for r1 in h_triplets:
#     for r2 in h_triplets:
#         if set(r1) & set(r2):
#             continue  # no overlap
        
#         # Check vertical connections (need at least 2 for fold-unfold)
#         vert_conns = []
#         for i, v1 in enumerate(r1):
#             for j, v2 in enumerate(r2):
#                 if (min(v1,v2), max(v1,v2)) in v2v:
#                     vert_conns.append((i, j, v1, v2))
        
#         if len(vert_conns) < 2:
#             continue
        
#         # Check for 3rd row
#         for r3 in h_triplets:
#             if set(r3) & set(r1) or set(r3) & set(r2):
#                 continue
            
#             vert_conns_23 = []
#             for i, v2 in enumerate(r2):
#                 for j, v3 in enumerate(r3):
#                     if (min(v2,v3), max(v2,v3)) in v2v:
#                         vert_conns_23.append((i, j, v2, v3))
            
#             if len(vert_conns_23) < 2:
#                 continue
            
#             # We have a candidate! Score by total vertical connections
#             score = len(vert_conns) + len(vert_conns_23)
#             if score > best_score:
#                 best_score = score
#                 best_layout = (r1, r2, r3, vert_conns, vert_conns_23)

# if not best_layout:
#     print("No valid 3x3 layout found!")
# else:
#     r1, r2, r3, vc12, vc23 = best_layout
#     print(f"\n=== Best 3x3 Layout (score={best_score}) ===")
#     print(f"  Row 0: D0={r1[0]}, D1={r1[1]}, D2={r1[2]}")
#     print(f"  Row 1: D3={r2[0]}, D4={r2[1]}, D5={r2[2]}")
#     print(f"  Row 2: D6={r3[0]}, D7={r3[1]}, D8={r3[2]}")
#     print(f"  Vertical R0-R1: {vc12}")
#     print(f"  Vertical R1-R2: {vc23}")
    
#     data = {0:r1[0], 1:r1[1], 2:r1[2], 3:r2[0], 4:r2[1], 5:r2[2], 6:r3[0], 7:r3[1], 8:r3[2]}
    
#     # Step 5: Full CX verification for fold-unfold scheme
#     print(f"\n=== Full CX Verification ===")
    
#     # Horizontal bridges
#     h_bridges = {}
#     for i, j in [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)]:
#         p1, p2 = data[i], data[j]
#         key = (min(p1,p2), max(p1,p2))
#         if key in v2v:
#             h_bridges[(i,j)] = v2v[key]
#             print(f"  H bridge D{i}({p1})-D{j}({p2}): link {v2v[key]} ✅")
#         else:
#             print(f"  H bridge D{i}({p1})-D{j}({p2}): NONE ❌")
    
#     # Vertical bridges
#     v_bridges = {}
#     for i, j in [(0,3),(1,4),(2,5),(3,6),(4,7),(5,8)]:
#         p1, p2 = data[i], data[j]
#         key = (min(p1,p2), max(p1,p2))
#         if key in v2v:
#             v_bridges[(i,j)] = v2v[key]
#             print(f"  V bridge D{i}({p1})-D{j}({p2}): link {v2v[key]} ✅")
#         else:
#             print(f"  V bridge D{i}({p1})-D{j}({p2}): NONE (fold needed)")
    
#     # Stabilizer measurement feasibility
#     x_stabs = [[0,1,3,4], [4,5,7,8], [3,6], [2,5]]
#     z_stabs = [[1,2,4,5], [3,4,6,7], [0,1], [7,8]]
    
#     print(f"\n=== Stabilizer Measurement Plan ===")
#     for typ, stabs in [("Z", z_stabs), ("X", x_stabs)]:
#         for idx, stab in enumerate(stabs):
#             phys = [data[d] for d in stab]
#             if len(stab) == 2:
#                 p1, p2 = data[stab[0]], data[stab[1]]
#                 key = (min(p1,p2), max(p1,p2))
#                 bridge = v2v.get(key, None)
#                 if bridge:
#                     print(f"  {typ}{idx} {stab}: weight-2, bridge={bridge} ✅")
#                 else:
#                     # Check relay path
#                     common_2hop = set()
#                     for n1 in adj[p1]:
#                         for n2 in adj[n1]:
#                             if n2 in adj[p2] or any(n3 in adj[p2] for n3 in adj[n2]):
#                                 common_2hop.add(n1)
#                     print(f"  {typ}{idx} {stab}: weight-2, NO direct bridge, relay needed via {common_2hop or 'UNKNOWN'}")
#             else:
#                 # Weight-4: fold to weight-2, then measure
#                 # Find which pairs can fold
#                 foldable = []
#                 for i in range(len(stab)):
#                     for j in range(i+1, len(stab)):
#                         p1, p2 = data[stab[i]], data[stab[j]]
#                         key = (min(p1,p2), max(p1,p2))
#                         if key in v2v:
#                             foldable.append((stab[i], stab[j], v2v[key]))
                
#                 # After fold, remaining weight-2 pair
#                 if len(foldable) >= 2:
#                     # Pick fold pairs that reduce to weight-2
#                     print(f"  {typ}{idx} {stab}: weight-4, folds={foldable}")
#                     # Check if folded pair has common ancilla
#                     fold1, fold2 = foldable[0], foldable[1]
#                     # After folding fold1[0]→fold1[1] and fold2[0]→fold2[1]
#                     # Remaining pair is fold1[1] and fold2[1]
#                     rem1, rem2 = data[fold1[1]], data[fold2[1]]
#                     key_rem = (min(rem1,rem2), max(rem1,rem2))
#                     anc = v2v.get(key_rem, None)
#                     if anc:
#                         print(f"    → fold to ({fold1[1]},{fold2[1]}), ancilla={anc} ✅")
#                     else:
#                         print(f"    → fold to ({fold1[1]},{fold2[1]}), NO ancilla ❌")
#                 else:
#                     print(f"  {typ}{idx} {stab}: weight-4, insufficient folds={foldable} ❌")



# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict

# service = QiskitRuntimeService(instance="Yonsei_internal")

# for name in ["ibm_boston", "ibm_pittsburgh", "ibm_fez", "ibm_kingston", "ibm_marrakesh"]:
#     try:
#         backend = service.backend(name)
#     except:
#         continue
    
#     edges = backend.coupling_map.get_edges()
#     adj = defaultdict(set)
#     for q1, q2 in edges:
#         adj[q1].add(q2)
#         adj[q2].add(q1)
#     degrees = {q: len(adj[q]) for q in adj}
#     vertices = sorted([q for q, d in degrees.items() if d == 3])
    
#     # vertex-vertex connections via link
#     v2v = {}
#     for v in vertices:
#         for link in adj[v]:
#             if degrees[link] == 2:
#                 for v2 in adj[link]:
#                     if v2 != v and degrees[v2] == 3 and v < v2:
#                         v2v[(v, v2)] = link
    
#     v_adj = defaultdict(set)
#     for (v1, v2) in v2v:
#         v_adj[v1].add(v2)
#         v_adj[v2].add(v1)
    
#     # Brute force: find 9 vertices forming a 3x3 grid
#     # Requirements: 
#     #  - 6 horizontal bridges (row connections)
#     #  - At least 4 vertical bridges (for fold-unfold)
#     best = None
#     best_score = -1
    
#     for v0 in vertices:
#         for v1 in v_adj[v0]:
#             for v2 in v_adj[v1]:
#                 if v2 == v0:
#                     continue
#                 row0 = [v0, v1, v2]
                
#                 for v3 in vertices:
#                     if v3 in row0:
#                         continue
#                     for v4 in v_adj[v3]:
#                         if v4 in row0 or v4 == v3:
#                             continue
#                         for v5 in v_adj[v4]:
#                             if v5 in row0 or v5 in [v3, v4]:
#                                 continue
#                             row1 = [v3, v4, v5]
                            
#                             # Count vertical connections between row0 and row1
#                             vert01 = 0
#                             for a in row0:
#                                 for b in row1:
#                                     if (min(a,b), max(a,b)) in v2v:
#                                         vert01 += 1
                            
#                             if vert01 < 2:
#                                 continue
                            
#                             for v6 in vertices:
#                                 if v6 in row0 + row1:
#                                     continue
#                                 for v7 in v_adj[v6]:
#                                     if v7 in row0 + row1 or v7 == v6:
#                                         continue
#                                     for v8 in v_adj[v7]:
#                                         if v8 in row0 + row1 or v8 in [v6, v7]:
#                                             continue
#                                         row2 = [v6, v7, v8]
                                        
#                                         vert12 = 0
#                                         for a in row1:
#                                             for b in row2:
#                                                 if (min(a,b), max(a,b)) in v2v:
#                                                     vert12 += 1
                                        
#                                         if vert12 < 2:
#                                             continue
                                        
#                                         score = vert01 + vert12
#                                         if score > best_score:
#                                             best_score = score
#                                             best = (name, row0, row1, row2, vert01, vert12)
#                                             if score >= 6:
#                                                 break
#                                     if best_score >= 6:
#                                         break
#                                 if best_score >= 6:
#                                     break
#                             if best_score >= 6:
#                                 break
#                         if best_score >= 6:
#                             break
#                     if best_score >= 6:
#                         break
#                 if best_score >= 6:
#                     break
#             if best_score >= 6:
#                 break
#         if best_score >= 6:
#             break
    
#     if best and best[0] == name:
#         _, r0, r1, r2, v01, v12 = best
#         print(f"\n{name}: FOUND layout (score={best_score})")
#         print(f"  Row0: {r0}")
#         print(f"  Row1: {r1}")
#         print(f"  Row2: {r2}")
#         print(f"  Vert R0-R1: {v01}, Vert R1-R2: {v12}")
#     else:
#         print(f"{name}: no layout found (best_score={best_score})")



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

# # 전체 구조를 있는 그대로 출력
# print("=== Full adjacency (degree-3 vertices only) ===")
# vertices = sorted([q for q, d in degrees.items() if d == 3])
# for v in vertices:
#     ns = sorted(adj[v])
#     # 각 neighbor의 degree와 그 너머 vertex
#     connections = []
#     for n in ns:
#         if degrees[n] == 2:
#             beyond = [q for q in adj[n] if q != v]
#             connections.append(f"--[{n}]-->{beyond[0]}(d{degrees[beyond[0]]})" if beyond else f"--[{n}]-->?")
#         else:
#             connections.append(f"--{n}(d{degrees[n]})")
#     print(f"  v{v}: {', '.join(connections)}")

# # 논문이 사용한 큐빗들 확인 (ibm_aachen/marrakesh 기준)
# # Fig 1(a)에서 (3,5) code의 배치를 찾아야 함
# print(f"\n=== Degree-1 boundary qubits ===")
# for q, d in sorted(degrees.items()):
#     if d == 1:
#         print(f"  q{q}: neighbor={sorted(adj[q])}")



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

# Heron heavy-hex chain 추출
# 수평 체인: vertex - link - link - link - vertex - link - link - link - vertex ...
# 대각선: vertex - link - vertex

# 먼저 전체 수평 체인을 추출
# 시작: degree-1 boundary qubit에서 출발
boundary = sorted([q for q, d in degrees.items() if d == 1])

def trace_chain(start, adj, degrees):
    """boundary부터 체인을 따라감"""
    chain = [start]
    visited = {start}
    current = start
    while True:
        found = False
        for n in sorted(adj[current]):
            if n not in visited:
                chain.append(n)
                visited.add(n)
                current = n
                found = True
                break
        if not found:
            break
    return chain

# 각 boundary에서 시작하는 체인
print("=== Horizontal chains from boundaries ===")
chains = []
for b in boundary:
    chain = trace_chain(b, adj, degrees)
    # 대각선 분기 전까지만 (첫 vertex에서 분기)
    print(f"  Start q{b}: first 15 qubits = {chain[:15]}")
    print(f"    degrees: {[degrees[q] for q in chain[:15]]}")

# 논문 Fig 1(a) 스타일로 배치 찾기
# Row를 수평 체인으로 정의
# 중심부의 넉넉한 영역 출력

# v41-v43-v45-v47-v49-v51-v53 수평 체인의 모든 큐빗
print("\n=== Chain between v41 and v53 (all qubits) ===")
# v41 neighbors: 40(d1), 42(d2), 36(d2)→v21
# v41 → 42 → v43 → 44 → v45 → 46 → v47 → 48 → v49 → 50 → v51 → 52 → v53
row_c = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
for q in row_c:
    ns = sorted(adj[q])
    label = "V" if degrees[q] == 3 else "L"
    print(f"  q{q}[{label}]: neighbors={ns}")

# 이 수평 체인에서 unit cell 식별:
# v41(A) - 42(Bridge) - v43(B=data) - 44(Bridge) - v45(C=ancilla?) - ...
# 또는 논문 방식:
# Data, Bridge, Ancilla_Z, Bridge, Ancilla_X, Bridge, Data
# 정확히는: 수평으로 7큐빗이 1 unit cell

print("\n=== Paper-style unit cell mapping ===")
print("Unit cell: A(data)-Bridge-B(anc_Z)-C(bridge)-D(anc_X)-E(bridge)-F(data)")
print()
print("Row C: v41..v53")
print(f"  Cell 1: A=v41, Bridge=42, B=v43, ignore")
print(f"  Looking at full structure:")
# 실제로는 Row B와 Row C 사이의 대각선이 unit cell의 수직 부분
# v21 → 36 → v41 (대각선 link)
# v23 → (v3은 위) and → v25 (수평)
# v25 → 37 → v45 (대각선)

print("\n=== Diagonal connections from Row B to Row C ===")
row_b_v = [21, 23, 25, 27, 29, 31, 33]
row_c_v = [41, 43, 45, 47, 49, 51, 53]
for vb in row_b_v:
    for n in adj[vb]:
        if degrees[n] == 2:
            for vc in adj[n]:
                if vc in row_c_v:
                    print(f"  v{vb} --[{n}]-- v{vc}")

print("\n=== Diagonal connections from Row C to Row D ===")
row_d_v = [61, 63, 65, 67, 69, 71, 73]
for vc in row_c_v:
    for n in adj[vc]:
        if degrees[n] == 2:
            for vd in adj[n]:
                if vd in row_d_v:
                    print(f"  v{vc} --[{n}]-- v{vd}")