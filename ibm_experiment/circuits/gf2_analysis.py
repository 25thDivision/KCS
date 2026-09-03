#!/usr/bin/env python3
"""
GF(2) characterization: find ALL deterministic parity constraints among
(18 syndrome bits + 9 data bits) of the noiseless 2-cycle depth-7 circuit.
A valid QEC memory circuit must supply enough deterministic constraints to
detect errors; their count and structure tell us exactly what is usable.
"""
import numpy as np
from qiskit_aer import AerSimulator
from ibm_experiment.circuits.heavyhex_depth7 import HeavyHexSurfaceCode

def sample_bits(num_cycles=2, dd=True, initial_state=0, shots=6000):
    sc = HeavyHexSurfaceCode(num_cycles=num_cycles, dd=dd)
    qc = sc.build_circuit(initial_state=initial_state)
    res = AerSimulator().run(qc, shots=shots).result().get_counts()
    rows = []
    for bs, c in res.items():
        d, s = bs.split()
        vec = [int(b) for b in s[::-1]] + [int(b) for b in d[::-1]]
        rows.append(vec)
    return np.array(sorted({tuple(r) for r in rows}), dtype=np.uint8)  # unique outcomes

def gf2_rank_and_nullspace(M):
    """Rank of rows of M over GF(2), and nullspace basis of M^T x = 0 i.e.
    vectors v with M @ v = 0 mod 2 (parity checks satisfied by all rows)."""
    M = M.copy() % 2
    n_rows, n_cols = M.shape
    # We want v such that every row r satisfies r·v = 0. Since a constraint can
    # also be affine (r·v = 1 for all rows), append a constant-1 column.
    A = np.hstack([M, np.ones((n_rows, 1), dtype=np.uint8)])
    # Nullspace of A over GF(2): solve A v' = 0 where v' = (v | c)
    A = A % 2
    m, n = A.shape
    R = A.copy()
    pivots = []
    row = 0
    for col in range(n):
        sel = None
        for r in range(row, m):
            if R[r, col]:
                sel = r; break
        if sel is None:
            continue
        R[[row, sel]] = R[[sel, row]]
        for r in range(m):
            if r != row and R[r, col]:
                R[r] ^= R[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    rank = len(pivots)
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, pc in enumerate(pivots):
            if i < row and R[i, f]:
                v[pc] = 1
        basis.append(v)
    return rank, basis

BIT_NAMES = (["c0." + n for n in ["Z1","X1","Xbr","Zbt","Z2","X2","Xbl","Zbb"]] +
             ["c1." + n for n in ["Z1","X1","Xbr","Zbt","Z2","X2","Xbl","Zbb"]] +
             ["fin.d16","fin.d17"] + [f"d{i}" for i in range(9)])

def describe(v):
    terms = [BIT_NAMES[i] for i in range(27) if v[i]]
    aff = " = 1" if v[27] else " = 0"
    return " ^ ".join(terms) + aff

if __name__ == "__main__":
    U = sample_bits()
    print(f"unique noiseless outcomes: {U.shape[0]} (dim of outcome space over GF2 <= rank)")
    # shift by one outcome so the affine space passes through origin, get linear dim
    base = U[0]
    D = (U ^ base)
    rank, _ = gf2_rank_and_nullspace(np.vstack([D, np.zeros(27, dtype=np.uint8)]))
    # rank of difference set = dimension of random subspace
    Dr = D.copy() % 2
    # gf2 row-rank of D
    M = Dr.copy()
    m, n = M.shape
    r = 0
    for col in range(n):
        sel = None
        for rr in range(r, m):
            if M[rr, col]:
                sel = rr; break
        if sel is None: continue
        M[[r, sel]] = M[[sel, r]]
        for rr in range(m):
            if rr != r and M[rr, col]:
                M[rr] ^= M[r]
        r += 1
    print(f"dimension of random outcome subspace: {r} / 27 bits")
    print(f"=> number of independent deterministic parity constraints: {27 - r}")

    _, basis = gf2_rank_and_nullspace(U)
    print(f"\ndeterministic constraints found: {len(basis)}")
    for v in basis:
        print("  " + describe(v))
