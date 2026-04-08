from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from collections import defaultdict

service = QiskitRuntimeService(instance="Yonsei_internal")
backend = service.backend("ibm_boston")
edges = backend.coupling_map.get_edges()

adj = defaultdict(set)
for q1, q2 in edges:
    adj[q1].add(q2)
    adj[q2].add(q1)

# ============================================================
# Step 1: Heron Layout (Eagle layout shifted by +1)
# ============================================================
# Eagle: D0=22,D1=24,D2=26,D3=41,D4=43,D5=45,D6=60,D7=62,D8=64
# Heron: D0=23,D1=25,D2=27,D3=43,D4=45,D5=47,D6=63,D7=65,D8=67

data = {0:23, 1:25, 2:27, 3:43, 4:45, 5:47, 6:63, 7:65, 8:67}
D = data

# Ancilla
z_anc = {0: 37, 1: 56, 2: 24, 3: 66}  # Z0,Z1,Z2,Z3
x_anc = {0: 37, 1: 57, 2: 56, 3: 38}  # X0,X1,X2,X3

# Fold bridges
fold_bridges = {
    "Z0_D2D1": (27, 26, 25),   # D2→D1 via 26
    "Z0_D5D4": (47, 46, 45),   # D5→D4 via 46
    "Z1_D4D3": (45, 44, 43),   # D4→D3 via 44
    "Z1_D7D6": (65, 64, 63),   # D7→D6 via 64
    "X0_D1D0": (25, 24, 23),   # D1→D0 via 24
    "X0_D4D3": (45, 44, 43),   # D4→D3 via 44
    "X1_D5D4": (47, 46, 45),   # D5→D4 via 46
    "X1_D8D7": (67, 66, 65),   # D8→D7 via 66
}

# X3 relay
x3_relay_d2 = [29, 28]   # anc38 → 29 → 28 → D2(27)
x3_relay_d5 = [49, 48]   # anc38 → 49 → 48 → D5(47)

# ============================================================
# Step 2: Verify ALL CX connections
# ============================================================
print("=== CX Verification ===")
all_ok = True

# Horizontal fold bridges
for label, (src, bridge, tgt) in fold_bridges.items():
    ok1 = bridge in adj[src]
    ok2 = tgt in adj[bridge]
    status = "✅" if (ok1 and ok2) else "❌"
    if not (ok1 and ok2): all_ok = False
    print(f"  {status} {label}: {src}→{bridge}({ok1})→{tgt}({ok2})")

# Ancilla direct measurement CX
meas_pairs = [
    ("Z0 anc37←D1", 25, 37), ("Z0 anc37←D4", 45, 37),
    ("Z1 anc56←D3", 43, 56), ("Z1 anc56←D6", 63, 56),
    ("Z2 anc24←D0", 23, 24), ("Z2 anc24←D1", 25, 24),
    ("Z3 anc66←D7", 65, 66), ("Z3 anc66←D8", 67, 66),
    ("X0 anc37←D1", 25, 37), ("X0 anc37←D4", 45, 37),
    ("X1 anc57←D5", 47, 57), ("X1 anc57←D8", 67, 57),
    ("X2 anc56←D3", 43, 56), ("X2 anc56←D6", 63, 56),
]
for label, q1, q2 in meas_pairs:
    ok = q2 in adj[q1]
    if not ok: all_ok = False
    print(f"  {'✅' if ok else '❌'} {label}: CX({q1},{q2})")

# X3 relay
relay_path_d2 = [38] + x3_relay_d2 + [27]
relay_path_d5 = [38] + x3_relay_d5 + [47]
for label, path in [("X3 relay→D2", relay_path_d2), ("X3 relay→D5", relay_path_d5)]:
    ok = all(path[i+1] in adj[path[i]] for i in range(len(path)-1))
    if not ok: all_ok = False
    print(f"  {'✅' if ok else '❌'} {label}: {' → '.join(str(q) for q in path)}")

print(f"\n  ALL OK: {all_ok}")

if not all_ok:
    print("ABORTING - fix layout first!")
    exit()

# ============================================================
# Step 3: Build Circuit
# ============================================================
print("\n=== Building Circuit ===")

n_qubits = max(max(adj.keys()), max(data.values())) + 1
all_physical = sorted(set(
    list(data.values()) + list(z_anc.values()) + list(x_anc.values()) +
    [26, 46, 44, 64, 24, 66] +  # fold bridges
    [28, 29, 48, 49]              # relay
))

num_cycles = 3
num_stabs = 8
cr_syn = f"syn"
cr_data = f"data"

qc = QuantumCircuit(n_qubits)
# syn_reg = qc.add_register(QuantumCircuit(num_stabs * num_cycles).cregs[0] if False else None)

# Rebuild with proper classical registers
from qiskit import ClassicalRegister
qc = QuantumCircuit(n_qubits)
cr_s = ClassicalRegister(num_stabs * num_cycles, 'syn')
cr_d = ClassicalRegister(9, 'data')
qc.add_register(cr_s)
qc.add_register(cr_d)

# Reset all used qubits
for q in all_physical:
    qc.reset(q)

def bridge_cx(qc, ctrl, tgt, bridge):
    qc.cx(ctrl, bridge)
    qc.cx(bridge, tgt)
    qc.cx(ctrl, bridge)

def relay_cx(qc, ctrl, tgt, relays):
    chain = [ctrl] + relays + [tgt]
    for i in range(len(chain)-1):
        qc.cx(chain[i], chain[i+1])
    for i in range(len(chain)-2, 0, -1):
        qc.cx(chain[i-1], chain[i])

for cycle in range(num_cycles):
    so = cycle * num_stabs  # syndrome offset
    qc.barrier()

    # === Z half-round ===
    # Z0: fold D2→D1, D5→D4, measure on anc37
    bridge_cx(qc, 27, 25, 26)
    bridge_cx(qc, 47, 45, 46)
    qc.cx(25, 37); qc.cx(45, 37)
    qc.measure(37, cr_s[so+0])
    bridge_cx(qc, 27, 25, 26)
    bridge_cx(qc, 47, 45, 46)

    # Z1: fold D4→D3, D7→D6, measure on anc56
    bridge_cx(qc, 45, 43, 44)
    bridge_cx(qc, 65, 63, 64)
    qc.cx(43, 56); qc.cx(63, 56)
    qc.measure(56, cr_s[so+1])
    bridge_cx(qc, 45, 43, 44)
    bridge_cx(qc, 65, 63, 64)

    # Z2: boundary D0,D1 on anc24
    qc.cx(23, 24); qc.cx(25, 24)
    qc.measure(24, cr_s[so+2])

    # Z3: boundary D7,D8 on anc66
    qc.cx(65, 66); qc.cx(67, 66)
    qc.measure(66, cr_s[so+3])

    qc.barrier()

    # Reset shared ancilla
    for a in [37, 56, 24, 66]:
        qc.reset(a)

    # === X half-round ===
    # X0: fold D1→D0, D4→D3, measure on anc37
    bridge_cx(qc, 25, 23, 24)
    bridge_cx(qc, 45, 43, 44)
    qc.h(37)
    qc.cx(37, 25); qc.cx(37, 45)
    qc.h(37)
    qc.measure(37, cr_s[so+4])
    bridge_cx(qc, 25, 23, 24)
    bridge_cx(qc, 45, 43, 44)

    # X1: fold D5→D4, D8→D7, measure on anc57
    bridge_cx(qc, 47, 45, 46)
    bridge_cx(qc, 67, 65, 66)
    qc.h(57)
    qc.cx(57, 47); qc.cx(57, 67)
    qc.h(57)
    qc.measure(57, cr_s[so+5])
    bridge_cx(qc, 47, 45, 46)
    bridge_cx(qc, 67, 65, 66)

    # X2: boundary D3,D6 on anc56
    qc.h(56)
    qc.cx(56, 43); qc.cx(56, 63)
    qc.h(56)
    qc.measure(56, cr_s[so+6])

    # X3: relay D2,D5 on anc38
    qc.h(38)
    relay_cx(qc, 38, 27, [29, 28])
    relay_cx(qc, 38, 47, [49, 48])
    qc.h(38)
    qc.measure(38, cr_s[so+7])

# Final data measurement
qc.barrier()
for i in range(9):
    qc.measure(data[i], cr_d[i])

ops = qc.count_ops()
cx_count = ops.get('cx', 0)
print(f"Pre-transpile: depth={qc.depth()}, CX={cx_count}")

# ============================================================
# Step 4: Transpile for ibm_boston
# ============================================================
print("\n=== Transpile Comparison ===")
for opt in [0, 1]:
    tqc = transpile(qc, backend=backend, optimization_level=opt)
    tops = tqc.count_ops()
    two_q = sum(tops.get(g,0) for g in ['cx','ecr','cz'])
    swaps = tops.get('swap', 0)
    print(f"  opt={opt}: depth={tqc.depth()}, 2q={two_q}, swaps={swaps}")

# ============================================================
# Step 5: AerSimulator verification
# ============================================================
print("\n=== AerSimulator Verification ===")
sim = AerSimulator(n_qubits=n_qubits)
result = sim.run(qc, shots=1000).result()
counts = result.get_counts()

logical_z_correct = 0
total = sum(counts.values())
for bitstr, count in counts.items():
    parts = bitstr.split()
    data_bits = parts[0] if len(parts) >= 2 else bitstr[-9:]
    d0 = int(data_bits[-1])
    d3 = int(data_bits[-4])
    d6 = int(data_bits[-7])
    logical_z = (d0 + d3 + d6) % 2
    if logical_z == 0:
        logical_z_correct += count

print(f"  Logical Z = 0: {logical_z_correct}/{total} = {logical_z_correct/total*100:.1f}%")
print(f"  {'✅ PASS' if logical_z_correct == total else '❌ FAIL'}")

print(f"\n=== Summary ===")
print(f"  Backend: ibm_boston (Heron R3, 156 qubits)")
print(f"  Layout: D=[23,25,27,43,45,47,63,65,67]")
print(f"  Pre-transpile: depth={qc.depth()}, CX={cx_count}")
print(f"  Ready for QPU: {'YES' if logical_z_correct == total else 'NO'}")