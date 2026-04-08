#!/usr/bin/env python3
"""
Fixed Heavy-Hex Layout Analyzer v2
Row classification by qubit number range (20-qubit bands).
"""
from qiskit_ibm_runtime import QiskitRuntimeService
from collections import defaultdict
import json

service = QiskitRuntimeService(instance="Yonsei_internal")
backend = service.backend("ibm_boston")
edges = backend.coupling_map.get_edges()

adj = defaultdict(set)
for q1, q2 in edges:
    adj[q1].add(q2)
    adj[q2].add(q1)

deg = {q: len(adj[q]) for q in range(backend.num_qubits)}
vertices = sorted([q for q in range(backend.num_qubits) if deg.get(q,0) == 3])

# Row assignment by qubit number bands (20-qubit wide)
rows = defaultdict(list)
for v in vertices:
    rows[v // 20].append(v)
rows = {k: sorted(v) for k, v in sorted(rows.items())}

print("=== ROWS ===")
row_list = []
for band, vs in rows.items():
    print(f"  Row {len(row_list)} (band {band}): {vs}")
    row_list.append(vs)

# Vertex-to-row mapping
v2r = {}
for ri, vs in enumerate(row_list):
    for v in vs:
        v2r[v] = ri

# Classify each vertex's 3 link-neighbors as horizontal or vertical
print("\n=== VERTICAL CONNECTIONS ===")
vert = {}  # vertex -> {link, target, dir}
hlinks_per_vertex = defaultdict(list)

for v in vertices:
    ri = v2r[v]
    for lk in adj[v]:
        if deg[lk] != 2:
            continue
        other = [n for n in adj[lk] if n != v]
        if not other or deg[other[0]] != 3:
            continue
        ov = other[0]
        if v2r.get(ov) == ri:
            hlinks_per_vertex[v].append((lk, ov))
        else:
            d = "DOWN" if v2r[ov] > ri else "UP"
            vert[v] = {"link": lk, "target": ov, "dir": d,
                       "from_row": ri, "to_row": v2r[ov]}

for ri, vs in enumerate(row_list):
    print(f"\n  Row {ri}: {vs}")
    for v in vs:
        h = [(lk, ov) for lk, ov in hlinks_per_vertex[v]]
        vinfo = vert.get(v)
        vstr = f"  {vinfo['dir']} link{vinfo['link']}→V{vinfo['target']}(row{vinfo['to_row']})" if vinfo else "  (none)"
        hstr = ", ".join(f"link{lk}→V{ov}" for lk, ov in h)
        print(f"    V{v}: H=[{hstr}] V=[{vstr}]")

# Find (3,3) patches: 3 consecutive vertices per row, 3 adjacent rows
print("\n=== PATCH SEARCH ===")

def get_consec_triples(vs):
    """Find all horizontally-connected 3-vertex chains in a row."""
    triples = []
    for i in range(len(vs)):
        for j in range(i+1, len(vs)):
            # Check v_i - v_j connected via link
            lk_ij = None
            for lk in adj[vs[i]]:
                if deg[lk] == 2 and vs[j] in adj[lk]:
                    lk_ij = lk; break
            if lk_ij is None:
                continue
            for k in range(j+1, len(vs)):
                lk_jk = None
                for lk in adj[vs[j]]:
                    if deg[lk] == 2 and vs[k] in adj[lk]:
                        lk_jk = lk; break
                if lk_jk is not None:
                    triples.append((vs[i], vs[j], vs[k], lk_ij, lk_jk))
    return triples

patches = []
for r1 in range(len(row_list)):
    for r2 in range(r1+1, len(row_list)):
        for r3 in range(r2+1, len(row_list)):
            t1s = get_consec_triples(row_list[r1])
            t2s = get_consec_triples(row_list[r2])
            t3s = get_consec_triples(row_list[r3])
            for t1 in t1s:
                for t2 in t2s:
                    for t3 in t3s:
                        s1 = set(t1[:3]); s2 = set(t2[:3]); s3 = set(t3[:3])
                        # Count vertical connections R1↔R2
                        c12 = [(v, vert[v]["link"], vert[v]["target"])
                               for v in s1 | s2
                               if v in vert and (
                                   (v in s1 and vert[v]["target"] in s2) or
                                   (v in s2 and vert[v]["target"] in s1))]
                        # Deduplicate (each pair appears twice)
                        c12_pairs = set()
                        c12_dedup = []
                        for src, lk, tgt in c12:
                            pair = tuple(sorted([src, tgt]))
                            if pair not in c12_pairs:
                                c12_pairs.add(pair)
                                c12_dedup.append((src, lk, tgt))
                        # Count vertical connections R2↔R3
                        c23 = [(v, vert[v]["link"], vert[v]["target"])
                               for v in s2 | s3
                               if v in vert and (
                                   (v in s2 and vert[v]["target"] in s3) or
                                   (v in s3 and vert[v]["target"] in s2))]
                        c23_pairs = set()
                        c23_dedup = []
                        for src, lk, tgt in c23:
                            pair = tuple(sorted([src, tgt]))
                            if pair not in c23_pairs:
                                c23_pairs.add(pair)
                                c23_dedup.append((src, lk, tgt))

                        if len(c12_dedup) >= 2 and len(c23_dedup) >= 2:
                            patches.append({
                                "rows": [r1, r2, r3],
                                "r1": list(t1[:3]), "r2": list(t2[:3]), "r3": list(t3[:3]),
                                "h1": [t1[3], t1[4]], "h2": [t2[3], t2[4]], "h3": [t3[3], t3[4]],
                                "v12": c12_dedup, "v23": c23_dedup,
                                "n_vert": len(c12_dedup) + len(c23_dedup),
                            })

patches.sort(key=lambda p: -p["n_vert"])
print(f"\nFound {len(patches)} patches")

for i, p in enumerate(patches[:5]):
    print(f"\n--- Patch {i+1} (verts={p['n_vert']}) rows={p['rows']} ---")
    print(f"  R{p['rows'][0]}: {p['r1']}  hlinks={p['h1']}")
    print(f"  R{p['rows'][1]}: {p['r2']}  hlinks={p['h2']}")
    print(f"  R{p['rows'][2]}: {p['r3']}  hlinks={p['h3']}")
    print(f"  V R1↔R2: {p['v12']}")
    print(f"  V R2↔R3: {p['v23']}")
    d = p['r1'] + p['r2'] + p['r3']
    
    # Stabilizer feasibility check
    # Need vertical links for: d1↔d4, d2↔d5, d3↔d6, d4↔d7, d0↔d3, d5↔d8
    all_v = {}
    for src, lk, tgt in p['v12'] + p['v23']:
        all_v[(src, tgt)] = lk
        all_v[(tgt, src)] = lk
    
    d0,d1,d2 = p['r1']; d3,d4,d5 = p['r2']; d6,d7,d8 = p['r3']
    checks = {
        "Z0(d1-d4)": all_v.get((d1,d4)),
        "Z1(d2-d5)": all_v.get((d2,d5)),  # or reversed orientation
        "X0(d3-d6)": all_v.get((d3,d6)),
        "X1(d4-d7)": all_v.get((d4,d7)),
        "Zbnd(d0-d3)": all_v.get((d0,d3)),
        "Zbnd(d5-d8)": all_v.get((d5,d8)),
    }
    ok = all(v is not None for v in checks.values())
    for name, lk in checks.items():
        print(f"    {name}: link={lk} {'✓' if lk else '✗'}")
    print(f"  {'✅ ALL VALID' if ok else '⚠️ MISSING LINKS'}")

# JSON output
print("\n=== JSON ===")
out = {
    "rows": row_list,
    "vertical": {str(v): vert[v] for v in sorted(vert)},
    "best_patches": [{
        "data": sorted(p['r1']+p['r2']+p['r3']),
        "by_row": [p['r1'], p['r2'], p['r3']],
        "row_idx": p['rows'],
        "hlinks": p['h1']+p['h2']+p['h3'],
        "vlinks_12": p['v12'], "vlinks_23": p['v23'],
        "valid": all(all_v.get((a,b)) is not None for a,b in
                     [(p['r1'][1],p['r2'][1]),(p['r1'][2],p['r2'][2]),
                      (p['r2'][0],p['r3'][0]),(p['r2'][1],p['r3'][1]),
                      (p['r1'][0],p['r2'][0]),(p['r2'][2],p['r3'][2])]),
    } for p in patches[:5]]
}
print(json.dumps(out, indent=2, default=str))