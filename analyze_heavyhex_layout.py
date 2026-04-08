#!/usr/bin/env python3
"""
Heavy-Hex Surface Code Layout Analyzer for ibm_boston (Heron R3)
================================================================
Benito et al. (Ref 25) + Vezvaee et al. 의 SWAP-based fold-unfold embedding을
ibm_boston 토폴로지에 매핑하기 위한 분석 스크립트.

사용법:
  python3 analyze_heavyhex_layout.py

출력:
  1) 전체 vertex/link 분류
  2) 수직 연결 방향 (↑/↓) 분석
  3) Valid (3,3) surface code patch 후보
  4) 각 patch의 stabilizer-to-qubit 매핑
  5) Check ancilla / bridge qubit 배정
"""

from qiskit_ibm_runtime import QiskitRuntimeService
from collections import defaultdict
import json

# ============================================================
# 1. 토폴로지 추출
# ============================================================
print("=" * 60)
print("  Step 1: Extracting ibm_boston topology")
print("=" * 60)

service = QiskitRuntimeService(instance="Yonsei_internal")
backend = service.backend("ibm_boston")
edges = backend.coupling_map.get_edges()

adj = defaultdict(set)
for q1, q2 in edges:
    adj[q1].add(q2)
    adj[q2].add(q1)

n_qubits = backend.num_qubits
degrees = {q: len(adj[q]) for q in range(n_qubits)}

vertices = sorted([q for q in range(n_qubits) if degrees.get(q, 0) == 3])
links = sorted([q for q in range(n_qubits) if degrees.get(q, 0) == 2])
unused = sorted([q for q in range(n_qubits) if degrees.get(q, 0) <= 1])

print(f"Total qubits: {n_qubits}")
print(f"Vertices (deg-3): {len(vertices)} → {vertices}")
print(f"Links (deg-2): {len(links)} → {links}")
print(f"Unused (deg≤1): {len(unused)} → {unused}")

# ============================================================
# 2. Vertex 행 분류 (horizontal chains)
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Classifying vertex rows")
print("=" * 60)

# vertex끼리 link를 통해 수평으로 연결된 chain 찾기
def find_horizontal_chains():
    """vertex들의 수평 체인을 찾습니다."""
    visited = set()
    chains = []

    for start_v in vertices:
        if start_v in visited:
            continue
        chain = [start_v]
        visited.add(start_v)

        # 양방향으로 수평 확장
        for direction in [1, -1]:  # right, left
            current = start_v
            while True:
                found = False
                for link_q in sorted(adj[current]):
                    if degrees[link_q] == 2:  # link qubit
                        for next_v in adj[link_q]:
                            if next_v != current and degrees[next_v] == 3 and next_v not in visited:
                                # 수평 연결인지 확인 (같은 row)
                                # 수직 연결은 vertex-link-vertex 패턴이지만
                                # link의 두 이웃이 모두 같은 row에 있으면 수평
                                if direction == 1:
                                    chain.append(next_v)
                                else:
                                    chain.insert(0, next_v)
                                visited.add(next_v)
                                current = next_v
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break
        chains.append(sorted(chain))

    return sorted(chains, key=lambda c: c[0])

rows = find_horizontal_chains()
print(f"\nFound {len(rows)} vertex rows:")
for i, row in enumerate(rows):
    print(f"  Row {i}: {row} ({len(row)} vertices)")

# ============================================================
# 3. 수직 연결 분석 (vertex ↔ link ↔ vertex)
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Vertical connections between rows")
print("=" * 60)

# 각 vertex의 수직 연결 찾기
# vertex의 3개 이웃 중 수평 link 2개와 수직 link 1개가 있음
def find_vertical_connections():
    """각 vertex의 수직 연결(link를 통해 다른 row의 vertex로)을 찾습니다."""
    # 먼저 각 vertex가 어떤 row에 있는지 매핑
    vertex_to_row = {}
    for row_idx, row in enumerate(rows):
        for v in row:
            vertex_to_row[v] = row_idx

    # 수평 link 찾기: 같은 row의 두 vertex를 연결하는 link
    horizontal_links = set()
    for link_q in links:
        ns = sorted(adj[link_q])
        if len(ns) == 2 and all(degrees[n] == 3 for n in ns):
            # 두 이웃이 모두 vertex이면
            if vertex_to_row.get(ns[0]) == vertex_to_row.get(ns[1]):
                horizontal_links.add(link_q)

    # 수직 link: vertex-link-vertex에서 두 vertex가 다른 row
    vertical_conns = {}  # vertex -> (link, other_vertex, direction)
    for v in vertices:
        row_idx = vertex_to_row[v]
        for link_q in adj[v]:
            if degrees[link_q] == 2 and link_q not in horizontal_links:
                # 이 link의 다른 쪽 vertex 찾기
                other = [n for n in adj[link_q] if n != v]
                if other and degrees[other[0]] == 3:
                    other_v = other[0]
                    other_row = vertex_to_row.get(other_v, -1)
                    if other_row > row_idx:
                        direction = "DOWN"
                    elif other_row < row_idx:
                        direction = "UP"
                    else:
                        direction = "SAME?"  # shouldn't happen
                    vertical_conns[v] = {
                        "link": link_q,
                        "target_vertex": other_v,
                        "direction": direction,
                        "from_row": row_idx,
                        "to_row": other_row,
                    }

    return vertical_conns, horizontal_links

vert_conns, horiz_links = find_vertical_connections()

print(f"\nHorizontal links: {sorted(horiz_links)}")
print(f"Vertical connections: {len(vert_conns)}")

for row_idx, row in enumerate(rows):
    print(f"\n  Row {row_idx}: {row}")
    for v in row:
        if v in vert_conns:
            vc = vert_conns[v]
            print(f"    V{v} → {vc['direction']} via link{vc['link']} → V{vc['target_vertex']} (row {vc['to_row']})")
        else:
            print(f"    V{v} → no vertical connection")

# ============================================================
# 4. (3,3) Surface Code Patch 후보 찾기
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Finding valid (3,3) surface code patches")
print("=" * 60)

def find_horizontal_link(v1, v2):
    """두 인접 vertex 사이의 수평 link qubit을 찾습니다."""
    common = adj[v1] & adj[v2]
    for q in common:
        if degrees[q] == 2 and q in horiz_links:
            return q
    return None

def find_valid_patches():
    """
    d=3 surface code를 위한 9개 data vertex patch를 찾습니다.
    
    Benito/Vezvaee 배치에서 unit cell은:
    A-bridge-B / Z-anc / C-bridge-D / X-anc / E-bridge-F
    
    3개 row에서 각 3개 vertex를 선택하되,
    수직 연결이 fold-unfold stabilizer를 지원해야 합니다.
    
    가능한 배치 패턴 (Vezvaee Fig 1(a) 참조):
    Row i:   V_a, V_b, V_c  (3 consecutive vertices, horizontally connected)
    Row i+1: V_d, V_e, V_f  (3 consecutive vertices)
    Row i+2: V_g, V_h, V_j  (3 consecutive vertices)
    
    수직 연결 요구사항:
    - Row i의 일부 vertex가 Row i+1의 vertex에 연결
    - Row i+1의 일부 vertex가 Row i+2의 vertex에 연결
    """
    patches = []
    
    # vertex_to_row 매핑
    vertex_to_row = {}
    for row_idx, row in enumerate(rows):
        for v in row:
            vertex_to_row[v] = row_idx
    
    # 연속 3-vertex 트리플렛을 각 row에서 추출
    def get_triples(row):
        """row 내에서 수평으로 연결된 3-vertex 조합"""
        triples = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                for k in range(j+1, len(row)):
                    v1, v2, v3 = row[i], row[j], row[k]
                    # 연속적으로 연결되어야 함 (v1-link-v2-link-v3)
                    link12 = find_horizontal_link(v1, v2)
                    link23 = find_horizontal_link(v2, v3)
                    if link12 is not None and link23 is not None:
                        triples.append((v1, v2, v3, link12, link23))
        return triples

    # 모든 row 쌍에서 패치 찾기
    for r1_idx in range(len(rows)):
        for r2_idx in range(r1_idx + 1, len(rows)):
            for r3_idx in range(r2_idx + 1, len(rows)):
                triples_r1 = get_triples(rows[r1_idx])
                triples_r2 = get_triples(rows[r2_idx])
                triples_r3 = get_triples(rows[r3_idx])
                
                for t1 in triples_r1:
                    for t2 in triples_r2:
                        for t3 in triples_r3:
                            v_r1 = t1[:3]
                            v_r2 = t2[:3]
                            v_r3 = t3[:3]
                            
                            # 수직 연결 확인
                            # Row1 → Row2 연결 수
                            conns_12 = []
                            for v in v_r1:
                                if v in vert_conns:
                                    vc = vert_conns[v]
                                    if vc["target_vertex"] in v_r2:
                                        conns_12.append((v, vc["link"], vc["target_vertex"]))
                            
                            # Row2 → Row3 연결 수
                            conns_23 = []
                            for v in v_r2:
                                if v in vert_conns:
                                    vc = vert_conns[v]
                                    if vc["target_vertex"] in v_r3:
                                        conns_23.append((v, vc["link"], vc["target_vertex"]))
                            
                            # 최소 2개의 수직 연결이 필요 (stabilizer coverage)
                            if len(conns_12) >= 2 and len(conns_23) >= 2:
                                patch = {
                                    "rows": [r1_idx, r2_idx, r3_idx],
                                    "data_r1": list(v_r1),
                                    "data_r2": list(v_r2),
                                    "data_r3": list(v_r3),
                                    "data_all": sorted(list(v_r1) + list(v_r2) + list(v_r3)),
                                    "hlinks_r1": [t1[3], t1[4]],
                                    "hlinks_r2": [t2[3], t2[4]],
                                    "hlinks_r3": [t3[3], t3[4]],
                                    "vconns_12": conns_12,
                                    "vconns_23": conns_23,
                                    "total_vertical": len(conns_12) + len(conns_23),
                                }
                                patches.append(patch)
    
    # 수직 연결 수로 정렬
    patches.sort(key=lambda p: -p["total_vertical"])
    return patches

patches = find_valid_patches()
print(f"\nFound {len(patches)} valid patch candidates")

# Top 10 출력
for i, p in enumerate(patches[:10]):
    print(f"\n--- Patch {i+1} (vertical conns: {p['total_vertical']}) ---")
    print(f"  Rows: {p['rows']}")
    print(f"  Row {p['rows'][0]} data: {p['data_r1']}")
    print(f"  Row {p['rows'][1]} data: {p['data_r2']}")
    print(f"  Row {p['rows'][2]} data: {p['data_r3']}")
    print(f"  Horizontal links R1: {p['hlinks_r1']}")
    print(f"  Horizontal links R2: {p['hlinks_r2']}")
    print(f"  Horizontal links R3: {p['hlinks_r3']}")
    print(f"  Vertical R1→R2: {p['vconns_12']}")
    print(f"  Vertical R2→R3: {p['vconns_23']}")
    print(f"  All 9 data qubits: {p['data_all']}")

# ============================================================
# 5. 최적 Patch의 Stabilizer 매핑
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Stabilizer mapping for best patches")
print("=" * 60)

if patches:
    for pi, best in enumerate(patches[:3]):
        print(f"\n{'='*40}")
        print(f"  Detailed analysis: Patch {pi+1}")
        print(f"{'='*40}")
        
        d_r1 = best["data_r1"]
        d_r2 = best["data_r2"]
        d_r3 = best["data_r3"]
        
        print(f"\nData qubits:")
        print(f"  Row {best['rows'][0]}: d0={d_r1[0]}, d1={d_r1[1]}, d2={d_r1[2]}")
        print(f"  Row {best['rows'][1]}: d3={d_r2[0]}, d4={d_r2[1]}, d5={d_r2[2]}")
        print(f"  Row {best['rows'][2]}: d6={d_r3[0]}, d7={d_r3[1]}, d8={d_r3[2]}")
        
        print(f"\nHorizontal links (bridges between adjacent data in same row):")
        print(f"  R1: {d_r1[0]}--[{best['hlinks_r1'][0]}]--{d_r1[1]}--[{best['hlinks_r1'][1]}]--{d_r1[2]}")
        print(f"  R2: {d_r2[0]}--[{best['hlinks_r2'][0]}]--{d_r2[1]}--[{best['hlinks_r2'][1]}]--{d_r2[2]}")
        print(f"  R3: {d_r3[0]}--[{best['hlinks_r3'][0]}]--{d_r3[1]}--[{best['hlinks_r3'][1]}]--{d_r3[2]}")
        
        print(f"\nVertical links:")
        for src, link, tgt in best["vconns_12"]:
            print(f"  R1→R2: V{src}--[link{link}]--V{tgt}")
        for src, link, tgt in best["vconns_23"]:
            print(f"  R2→R3: V{src}--[link{link}]--V{tgt}")
        
        # Stabilizer assignment 시도
        # d=3 rotated surface code stabilizers:
        # Weight-4: Z0(d0,d1,d3,d4), Z1(d1,d2,d4,d5), X0(d3,d4,d6,d7), X1(d4,d5,d7,d8)
        # Weight-2: Z_bnd1(d0,d3 or d6,d7), X_bnd1(d0,d1 or d2,d5), etc.
        # 정확한 stabilizer는 코드 orientation에 따라 다름
        
        print(f"\nProposed stabilizer assignment (standard rotated d=3):")
        print(f"  Weight-4 Z-stabilizers:")
        print(f"    Z0: d0={d_r1[0]}, d1={d_r1[1]}, d3={d_r2[0]}, d4={d_r2[1]}")
        print(f"    Z1: d1={d_r1[1]}, d2={d_r1[2]}, d4={d_r2[1]}, d5={d_r2[2]}")
        print(f"  Weight-4 X-stabilizers:")
        print(f"    X0: d3={d_r2[0]}, d4={d_r2[1]}, d6={d_r3[0]}, d7={d_r3[1]}")
        print(f"    X1: d4={d_r2[1]}, d5={d_r2[2]}, d7={d_r3[1]}, d8={d_r3[2]}")
        print(f"  Weight-2 boundary stabilizers:")
        print(f"    Z_bnd0: d0={d_r1[0]}, d3={d_r2[0]}")
        print(f"    Z_bnd1: d5={d_r2[2]}, d8={d_r3[2]}")
        print(f"    X_bnd0: d0={d_r1[0]}, d1={d_r1[1]}")
        print(f"    X_bnd1: d7={d_r3[1]}, d8={d_r3[2]}")
        
        # 각 stabilizer의 fold-unfold에 필요한 vertical link 확인
        print(f"\n  Fold-unfold feasibility:")
        vconn_dict_12 = {(src, tgt): link for src, link, tgt in best["vconns_12"]}
        vconn_dict_21 = {(tgt, src): link for src, link, tgt in best["vconns_12"]}
        vconn_dict_23 = {(src, tgt): link for src, link, tgt in best["vconns_23"]}
        vconn_dict_32 = {(tgt, src): link for src, link, tgt in best["vconns_23"]}
        all_vconns = {**vconn_dict_12, **vconn_dict_21, **vconn_dict_23, **vconn_dict_32}
        
        # Z0: fold d0→d1 (hlink), fold d3→d4 (hlink), measure Z(d1,d4) via vertical link
        z0_fold_ok = (d_r1[0], d_r1[1]) in [(d_r1[0], d_r1[1])]  # hlink always exists
        z0_measure = all_vconns.get((d_r1[1], d_r2[1]))
        print(f"    Z0: fold via hlinks ✓, measure d1-d4 vertical link = {z0_measure}")
        
        z1_measure = all_vconns.get((d_r1[2], d_r2[2]))
        print(f"    Z1: fold via hlinks ✓, measure d2-d5 vertical link = {z1_measure}")
        
        x0_measure = all_vconns.get((d_r2[0], d_r3[0]))
        print(f"    X0: fold via hlinks ✓, measure d3-d6 vertical link = {x0_measure}")
        
        x1_measure = all_vconns.get((d_r2[1], d_r3[1]))
        print(f"    X1: fold via hlinks ✓, measure d4-d7 vertical link = {x1_measure}")
        
        # Boundary stabilizers
        z_bnd0 = all_vconns.get((d_r1[0], d_r2[0]))
        z_bnd1 = all_vconns.get((d_r2[2], d_r3[2]))
        print(f"    Z_bnd0: d0-d3 vertical link = {z_bnd0}")
        print(f"    Z_bnd1: d5-d8 vertical link = {z_bnd1}")
        
        # 수평 boundary stabilizers는 hlink가 ancilla 역할
        print(f"    X_bnd0: d0-d1 hlink = {best['hlinks_r1'][0]}")
        print(f"    X_bnd1: d7-d8 hlink = {best['hlinks_r3'][1]}")
        
        # 모든 vertical link 있는지 확인
        needed_vlinks = [z0_measure, z1_measure, x0_measure, x1_measure, z_bnd0, z_bnd1]
        missing = [f"idx{i}" for i, v in enumerate(needed_vlinks) if v is None]
        if missing:
            print(f"\n  ⚠️  MISSING vertical links: {missing}")
            print(f"  → This patch needs alternative stabilizer orientation!")
        else:
            print(f"\n  ✅ ALL vertical links present! This patch is valid.")
            print(f"\n  Summary of all physical qubits needed:")
            all_data = sorted(d_r1 + d_r2 + d_r3)
            all_hlinks = best['hlinks_r1'] + best['hlinks_r2'] + best['hlinks_r3']
            all_vlinks = [v for v in needed_vlinks if v is not None]
            all_qubits = sorted(set(all_data + all_hlinks + all_vlinks))
            print(f"    Data (9): {all_data}")
            print(f"    H-links (6): {sorted(all_hlinks)}")
            print(f"    V-links/ancilla (6): {sorted(all_vlinks)}")
            print(f"    Total physical qubits: {len(all_qubits)}")
            print(f"    All qubits: {all_qubits}")

# ============================================================
# 6. JSON 출력 (Claude.ai에 붙여넣기용)
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: JSON summary (paste this back to Claude)")
print("=" * 60)

output = {
    "backend": "ibm_boston",
    "n_qubits": n_qubits,
    "n_vertices": len(vertices),
    "n_links": len(links),
    "vertices": vertices,
    "links": links,
    "rows": [list(r) for r in rows],
    "vertical_connections": {
        str(v): vert_conns[v] for v in sorted(vert_conns.keys())
    },
    "horizontal_links": sorted(horiz_links),
    "top_patches": [
        {
            "data": p["data_all"],
            "rows_used": p["rows"],
            "data_by_row": [p["data_r1"], p["data_r2"], p["data_r3"]],
            "hlinks": p["hlinks_r1"] + p["hlinks_r2"] + p["hlinks_r3"],
            "vconns_12": [(s, l, t) for s, l, t in p["vconns_12"]],
            "vconns_23": [(s, l, t) for s, l, t in p["vconns_23"]],
            "total_vertical": p["total_vertical"],
        }
        for p in patches[:5]
    ],
}

print(json.dumps(output, indent=2))