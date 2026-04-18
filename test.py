from qiskit_ibm_runtime import QiskitRuntimeService
from collections import defaultdict
import json

service = QiskitRuntimeService(instance="Yonsei_internal")
backend = service.backend("ibm_miami")

print(f"=== {backend.name} ===")
print(f"Num qubits: {backend.num_qubits}")

cm = backend.coupling_map
edges = sorted(cm.get_edges())
print(f"Num edges (directed): {len(edges)}")

degree = defaultdict(int)
for a, b in edges:
    degree[a] += 1
degree_dist = defaultdict(int)
for q, d in degree.items():
    degree_dist[d] += 1
print(f"Degree distribution: {dict(sorted(degree_dist.items()))}")

print("\n=== First 20 qubits and their neighbors ===")
for q in range(20):
    neighbors = sorted([b for a, b in edges if a == q])
    print(f"  Q{q}: {neighbors}")

undirected = set()
for a, b in edges:
    undirected.add((min(a, b), max(a, b)))
with open("ibm_miami_edges.json", "w") as f:
    json.dump(sorted(list(undirected)), f)
print(f"\nSaved {len(undirected)} undirected edges to ibm_miami_edges.json")