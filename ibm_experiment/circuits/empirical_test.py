#!/usr/bin/env python3
"""
Noiseless Aer simulation of the depth-7 heavy-hex circuit.
Checks, for each syndrome bit and useful XOR combinations, whether the
outcome is deterministic (as a valid stabilizer measurement chain must be
for Z-type checks in a Z-basis memory experiment).
"""
import numpy as np
from qiskit_aer import AerSimulator
from heavyhex_depth7 import HeavyHexSurfaceCode
# from heavyhex_surface_code_depth7 import HeavyHexSurfaceCode

SHOTS = 4000

def run(num_cycles, dd, initial_state=0):
    sc = HeavyHexSurfaceCode(num_cycles=num_cycles, dd=dd)
    qc = sc.build_circuit(initial_state=initial_state)
    sim = AerSimulator()
    res = sim.run(qc, shots=SHOTS).result().get_counts()
    n_syn = 8 * num_cycles + 2
    syn = []
    dat = []
    counts = []
    for bs, c in res.items():
        # qiskit: registers space-separated, later-added register first? creg order: syn then data -> bitstring "data syn"
        parts = bs.split()
        assert len(parts) == 2
        data_bits, syn_bits = parts[0], parts[1]
        syn.append([int(b) for b in syn_bits[::-1]])   # little-endian -> index order
        dat.append([int(b) for b in data_bits[::-1]])
        counts.append(c)
    return np.array(syn), np.array(dat), np.array(counts), sc

def frac_ones(bits, counts):
    tot = counts.sum()
    return (bits * counts).sum() / tot

BIT_NAMES = ["Z1", "X1", "Xb_right", "Zb_top", "Z2", "X2", "Xb_left", "Zb_bot"]

for dd in [False, True]:
    print(f"\n{'='*70}\n  dd={dd}, initial |0>_L, 2 cycles, noiseless, {SHOTS} shots\n{'='*70}")
    syn, dat, counts, sc = run(2, dd)
    n_syn = syn.shape[1]
    print(f"syndrome bits: {n_syn}")
    print(f"{'bit':>4} {'name':<22} {'P(1)':>8}  verdict")
    for b in range(n_syn):
        cyc = b // 8
        name = BIT_NAMES[b % 8] if b < 16 else "deferred_final"
        p = frac_ones(syn[:, b], counts)
        v = "DET-0" if p < 0.005 else ("DET-1" if p > 0.995 else ("RANDOM" if 0.45 < p < 0.55 else f"BIASED"))
        print(f"{b:>4} c{cyc}.{name:<20} {p:>8.3f}  {v}")

    # cycle-to-cycle XOR (software differencing) for each of the 8 bits
    print("\n  cycle-to-cycle XOR (bit ^ bit+8):")
    for b in range(8):
        x = syn[:, b] ^ syn[:, b + 8]
        p = frac_ones(x, counts)
        v = "DET-0" if p < 0.005 else ("DET-1" if p > 0.995 else ("RANDOM" if 0.45 < p < 0.55 else "BIASED"))
        print(f"    {BIT_NAMES[b]:<10} {p:>8.3f}  {v}")

    # data-qubit stabilizer/logical consistency at final measurement
    print("\n  final data-qubit checks:")
    for name, idxs in [("Z{0,1,3,4}", [0,1,3,4]), ("Z{1,2,4,5}", [1,2,4,5]),
                       ("Z{0,1}", [0,1]), ("Z{7,8}", [7,8]),
                       ("Z_L{0,3,6}", [0,3,6])]:
        par = np.bitwise_xor.reduce(dat[:, idxs], axis=1)
        p = frac_ones(par, counts)
        v = "DET-0" if p < 0.005 else ("DET-1" if p > 0.995 else ("RANDOM" if 0.45 < p < 0.55 else "BIASED"))
        print(f"    {name:<12} {p:>8.3f}  {v}")

    # consistency: last Z-syndrome vs final-data parity (projected stabilizer agreement)
    print("\n  last-cycle Z-bit vs final data parity XOR:")
    for b, idxs, name in [(8, [0,1,3,4], "Z1"), (12, [1,2,4,5], "Z2"),
                          (11, [0,1], "Zb_top"), (15, [7,8], "Zb_bot")]:
        par = np.bitwise_xor.reduce(dat[:, idxs], axis=1) ^ syn[:, b]
        p = frac_ones(par, counts)
        v = "DET-0" if p < 0.005 else ("DET-1" if p > 0.995 else ("RANDOM" if 0.45 < p < 0.55 else "BIASED"))
        print(f"    {name:<8}^data {p:>8.3f}  {v}")
