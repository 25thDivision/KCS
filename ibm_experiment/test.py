# from qiskit_ibm_runtime import QiskitRuntimeService
# from collections import defaultdict
# import json

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_miami")

# print(f"=== {backend.name} ===")
# print(f"Num qubits: {backend.num_qubits}")

# cm = backend.coupling_map
# edges = sorted(cm.get_edges())
# print(f"Num edges (directed): {len(edges)}")

# degree = defaultdict(int)
# for a, b in edges:
#     degree[a] += 1
# degree_dist = defaultdict(int)
# for q, d in degree.items():
#     degree_dist[d] += 1
# print(f"Degree distribution: {dict(sorted(degree_dist.items()))}")

# print("\n=== First 20 qubits and their neighbors ===")
# for q in range(20):
#     neighbors = sorted([b for a, b in edges if a == q])
#     print(f"  Q{q}: {neighbors}")

# undirected = set()
# for a, b in edges:
#     undirected.add((min(a, b), max(a, b)))
# with open("ibm_miami_edges.json", "w") as f:
#     json.dump(sorted(list(undirected)), f)
# print(f"\nSaved {len(undirected)} undirected edges to ibm_miami_edges.json")

# from qiskit_ibm_runtime import QiskitRuntimeService
# from utils.nighthawk_layout import select_best_patch

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_miami")

# for d in [3, 5]:
#     res = select_best_patch(backend, distance=d)
#     diag = res['diagnostics']
#     print(f"\nd={d}:")
#     print(f"  origin: {diag['patch_origin']}")
#     print(f"  avg CX error: {diag['avg_cx_error']:.4e}")
#     print(f"  avg readout error: {diag['avg_readout_error']:.4e}")
#     print(f"  data qubits: {res['data_qubits']}")
#     print(f"  ancilla qubits: {res['ancilla_qubits']}")

# from qiskit import transpile
# from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit

# for d in [3, 5]:
#     layout = select_best_patch(backend, distance=d)
#     sc = SurfaceCodeCircuit(distance=d, num_rounds=d, 
#                             physical_qubits=layout['physical_qubits'])
#     qc = sc.build_circuit(initial_state=0)
    
#     print(f"\n=== d={d} ===")
#     print(f"Pre-transpile: depth={qc.depth()}, ops={qc.count_ops()}")
    
#     for opt in [0, 1, 2, 3]:
#         tqc = transpile(qc, backend=backend, 
#                        initial_layout=layout['initial_layout'],
#                        optimization_level=opt)
#         ops = tqc.count_ops()
#         two_q = sum(ops.get(g, 0) for g in ['cx', 'ecr', 'cz'])
#         swap = ops.get('swap', 0)
#         print(f"  opt={opt}: depth={tqc.depth()}, 2q={two_q}, swap={swap}")

# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit import transpile
# from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
# from utils.nighthawk_layout import select_best_patch

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_miami")

# for d in [3, 5]:
#     layout = select_best_patch(backend, distance=d)
#     sc = SurfaceCodeCircuit(distance=d, num_rounds=d, 
#                             physical_qubits=layout['physical_qubits'])
#     qc = sc.build_circuit(initial_state=0)
    
#     print(f"\n=== d={d} ===")
#     print(f"Pre-transpile: depth={qc.depth()}, ops={dict(qc.count_ops())}")
    
#     for opt in [1, 2, 3]:
#         try:
#             tqc = transpile(qc, backend=backend, 
#                            initial_layout=layout['initial_layout'],
#                            optimization_level=opt)
#             ops = dict(tqc.count_ops())
#             two_q = sum(ops.get(g, 0) for g in ['cx', 'ecr', 'cz'])
#             swap = ops.get('swap', 0)
#             print(f"  opt={opt}: depth={tqc.depth()}, 2q={two_q}, swap={swap}, "
#                   f"ops={ops}")
#         except Exception as e:
#             print(f"  opt={opt}: FAILED — {e}")

# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
# from utils.nighthawk_layout import select_best_patch
# import qiskit

# print(f"Qiskit version: {qiskit.__version__}")

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_miami")

# # Nighthawk의 실제 native gate set (snapshot/store 제외)
# NIGHTHAWK_BASIS = ['cz', 'sx', 'rz', 'x', 'id', 'measure', 'reset', 'barrier', 'delay']

# for d in [3, 5]:
#     layout = select_best_patch(backend, distance=d)
#     sc = SurfaceCodeCircuit(distance=d, num_rounds=d,
#                             physical_qubits=layout['physical_qubits'])
#     qc = sc.build_circuit(initial_state=0)
    
#     print(f"\n=== d={d} ===")
#     print(f"Pre-transpile: depth={qc.depth()}, ops={dict(qc.count_ops())}")
    
#     for opt in [1, 2, 3]:
#         try:
#             pm = generate_preset_pass_manager(
#                 optimization_level=opt,
#                 basis_gates=NIGHTHAWK_BASIS,
#                 coupling_map=backend.coupling_map,
#                 initial_layout=layout['initial_layout'],
#             )
#             tqc = pm.run(qc)
#             ops = dict(tqc.count_ops())
#             two_q = sum(ops.get(g, 0) for g in ['cx', 'ecr', 'cz'])
#             swap = ops.get('swap', 0)
#             print(f"  opt={opt}: depth={tqc.depth()}, 2q={two_q}, swap={swap}, ops={ops}")
#         except Exception as e:
#             print(f"  opt={opt}: FAILED — {type(e).__name__}: {str(e)[:200]}")

# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
# from utils.nighthawk_layout import select_best_patch
# import qiskit

# print(f"Qiskit version: {qiskit.__version__}")

# service = QiskitRuntimeService(instance="Yonsei_internal")
# backend = service.backend("ibm_miami")

# # 표준 1Q/2Q gate만 포함. measure/reset/barrier/delay는 Qiskit이 자동 인식.
# NIGHTHAWK_GATES = ['cz', 'sx', 'rz', 'x', 'id']

# for d in [3, 5]:
#     layout = select_best_patch(backend, distance=d)
#     sc = SurfaceCodeCircuit(distance=d, num_rounds=d,
#                             physical_qubits=layout['physical_qubits'])
#     qc = sc.build_circuit(initial_state=0)
    
#     print(f"\n=== d={d} ===")
#     print(f"Pre-transpile: depth={qc.depth()}, ops={dict(qc.count_ops())}")
    
#     for opt in [1, 2, 3]:
#         try:
#             pm = generate_preset_pass_manager(
#                 optimization_level=opt,
#                 basis_gates=NIGHTHAWK_GATES,
#                 coupling_map=backend.coupling_map,
#                 initial_layout=layout['initial_layout'],
#             )
#             tqc = pm.run(qc)
#             ops = dict(tqc.count_ops())
#             two_q = sum(ops.get(g, 0) for g in ['cx', 'ecr', 'cz'])
#             swap = ops.get('swap', 0)
#             print(f"  opt={opt}: depth={tqc.depth()}, 2q={two_q}, swap={swap}, ops={ops}")
#         except Exception as e:
#             print(f"  opt={opt}: FAILED — {type(e).__name__}: {str(e)[:250]}")


from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(instance="Yonsei_internal")
backend = service.backend("ibm_miami")

# Circuit A: measure twice without reset
qc = QuantumCircuit(1, 2)
qc.h(0)
qc.measure(0, 0)        # m1
qc.measure(0, 1)        # m2 (same qubit, no reset between)

# Use our standard transpile settings
pm = generate_preset_pass_manager(
    optimization_level=2,
    basis_gates=['cz', 'sx', 'rz', 'x', 'id'],
    coupling_map=backend.coupling_map,
    initial_layout=[13],  # use a known-good qubit from our d=3 patch
)
tqc = pm.run(qc)

sampler = Sampler(mode=backend)
job = sampler.run([tqc], shots=100)
result = job.result()

counts = result[0].data.meas.get_counts() if hasattr(result[0].data, 'meas') else result[0].data.c.get_counts()

# Analyze: count m1==m2 vs m1!=m2
same = 0
diff = 0
for bitstr, cnt in counts.items():
    # bitstr is '00', '01', '10', '11'. Parsing: first bit is cbit[1], second is cbit[0]
    # or depends on endianness — need to verify once
    m2_bit, m1_bit = bitstr[0], bitstr[1]  # likely
    if m1_bit == m2_bit:
        same += cnt
    else:
        diff += cnt

print(f"m1 == m2: {same}/100")
print(f"m1 != m2: {diff}/100")
print()
print(f"Interpretation:")
print(f"  If no-reset works: m1==m2 should be ~100/100")
print(f"  If implicit reset: m1==m2 should be ~50/100 (m2 always 0, m1 random)")
print(f"  If decoherence dominates: m1==m2 somewhere between, e.g. 60-70/100")