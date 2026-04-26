"""
Nighthawk (ibm_miami) qubit selection for rotated surface code.

ibm_miami is a 120-qubit 4-neighbor square lattice with row-major indexing:
Q0..Q9 form the top row, Q10..Q19 the second row, etc. Its coupling map has
no diagonal edges, so Stim's canonical rotated_memory_z layout (which uses
sqrt(2) diagonal edges between data and ancilla in the native coord system)
cannot be dropped in directly.

We work around this by extracting Stim's canonical QUBIT_COORDS and applying
a 45-degree rotation (x, y) -> ((x+y)//2, (y-x)//2). After rotation all
data<->ancilla edges are orthogonal (distance 1) and the patch is an
axis-aligned (2d-1) x (2d-1) square embeddable in a 4-neighbor grid.

Stim qubit indices are mapped to Qiskit SurfaceCodeCircuit logical indices by:
  - data qubits: sorted Stim index -> 0..d^2-1 (Stim's data order already
    matches Qiskit's row-major order because Stim coords sort by (y, x))
  - ancillas: Stim ancilla is matched to Qiskit stabilizer with equal support
    (support sets are identical under the sorted data mapping; the
    Stim<->Qiskit X/Z label swap is a labeling difference handled downstream)

Logical indexing compatible with SurfaceCodeCircuit:
  - Data qubits: logical indices 0 .. d^2 - 1
  - Ancilla qubits: logical indices d^2 .. d^2 + num_stabilizers - 1
    (order matches SurfaceCodeCircuit.x_stabilizers + z_stabilizers)
"""

import os
import sys

import numpy as np
import stim
from typing import Dict, List, Tuple, Optional

_this_dir = os.path.dirname(os.path.abspath(__file__))
_ibm_dir = os.path.dirname(_this_dir)
if _ibm_dir not in sys.path:
    sys.path.insert(0, _ibm_dir)

from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit


# Row-major square lattice assumed for ibm_miami (Nighthawk).
NIGHTHAWK_GRID_SHAPE: Tuple[int, int] = (12, 10)  # (num_rows, num_cols)


def index_to_coord(idx: int,
                   grid_shape: Tuple[int, int] = NIGHTHAWK_GRID_SHAPE
                   ) -> Tuple[int, int]:
    _, ncols = grid_shape
    return (idx // ncols, idx % ncols)


def coord_to_index(row: int, col: int,
                   grid_shape: Tuple[int, int] = NIGHTHAWK_GRID_SHAPE) -> int:
    _, ncols = grid_shape
    return row * ncols + col


def _infer_grid_shape(num_qubits: int) -> Tuple[int, int]:
    """Return (rows, cols) for a row-major square lattice backend."""
    expected = NIGHTHAWK_GRID_SHAPE[0] * NIGHTHAWK_GRID_SHAPE[1]
    if num_qubits == expected:
        return NIGHTHAWK_GRID_SHAPE
    # Generic fallback: factor pair closest to square.
    best = (1, num_qubits)
    for r in range(1, int(np.sqrt(num_qubits)) + 2):
        if num_qubits % r == 0:
            c = num_qubits // r
            if abs(c - r) < abs(best[1] - best[0]):
                best = (r, c)
    return best


def _coupling_adjacency(edges: List[Tuple[int, int]]) -> Dict[int, set]:
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _extract_stim_canonical_layout(distance: int):
    """
    Parse a Stim-generated surface_code:rotated_memory_z circuit to extract
    the canonical embedding. Returns:
      coords:         {stim_qubit_idx: (x, y)} integer coordinates
      stim_data:      sorted list of Stim data qubit indices
      stim_ancilla:   list of Stim ancilla indices in first-MR order
      stim_support:   {stim_anc_idx: sorted list of Stim data indices it CXs with}
    """
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance, rounds=distance,
        before_round_data_depolarization=0.001,
    )

    coords: Dict[int, Tuple[int, int]] = {}
    for inst in circ:
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            for t in inst.targets_copy():
                coords[t.value] = (int(args[0]), int(args[1]))

    data_set = set()
    anc_order: List[int] = []
    anc_seen = set()
    for inst in circ.flattened():
        if inst.name == "DEPOLARIZE1":
            for t in inst.targets_copy():
                data_set.add(t.value)
        elif inst.name == "MR":
            for t in inst.targets_copy():
                if t.value not in anc_seen:
                    anc_order.append(t.value)
                    anc_seen.add(t.value)
    stim_data = sorted(data_set)
    stim_ancilla = [a for a in anc_order if a not in data_set]

    cx_src: Dict[int, set] = {}
    cx_tgt: Dict[int, set] = {}
    for inst in circ.flattened():
        if inst.name in ("CX", "CNOT", "ZCX", "CZ", "XCZ"):
            ts = [t.value for t in inst.targets_copy()]
            for i in range(0, len(ts), 2):
                s, t = ts[i], ts[i + 1]
                cx_src.setdefault(s, set()).add(t)
                cx_tgt.setdefault(t, set()).add(s)
    stim_support: Dict[int, List[int]] = {}
    for a in stim_ancilla:
        supp = (cx_src.get(a, set()) | cx_tgt.get(a, set())) & data_set
        stim_support[a] = sorted(supp)

    return coords, stim_data, stim_ancilla, stim_support


def _rotate_45(coords: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
    """Apply (x, y) -> ((x+y)//2, (y-x)//2) and normalize to non-negative origin."""
    raw = {q: ((x + y) // 2, (y - x) // 2) for q, (x, y) in coords.items()}
    min_r = min(p[0] for p in raw.values())
    min_c = min(p[1] for p in raw.values())
    return {q: (r - min_r, c - min_c) for q, (r, c) in raw.items()}


def _build_logical_to_grid(distance: int) -> Tuple[Dict[int, Tuple[int, int]], List[Tuple[int, Tuple[int, int], Tuple[int, int]]]]:
    """
    Build logical-qubit -> local grid-position mapping using the 45-degree
    rotation of Stim's canonical rotated_memory_z layout. Resulting embedding
    is native to a 4-neighbor square lattice: every CX edge has manhattan
    distance 1.

    Returns:
      logical_to_pos: dict mapping logical index -> (row, col) within a
        (2d-1) x (2d-1) local grid.
      cx_edges: list of (ancilla_logical, anc_pos, data_pos) tuples for each
        required CX adjacency, with anc_pos / data_pos given as grid positions.

    Logical indexing:
      - Data qubits: 0 .. d^2 - 1, ordered by Qiskit SurfaceCodeCircuit row-major.
      - Ancillas:    d^2 .. d^2 + num_stab - 1, ordered as x_stabilizers + z_stabilizers.

    Stim -> Qiskit mapping:
      - Data: sorted Stim index -> Qiskit logical index (Stim coord order
        sorts by y then x, which matches Qiskit row-major).
      - Ancilla: each Qiskit stabilizer is matched to the Stim ancilla whose
        CX support (expressed in Qiskit data indices) is set-equal to it.
        (Support sets coincide; only the X/Z label differs between Stim and
        Qiskit, and that labeling is handled by the measurement schedule.)

    Raises AssertionError if any CX edge is not 4-neighbor after rotation,
    or if any Qiskit stabilizer lacks a matching Stim ancilla.
    """
    d = distance
    sc = SurfaceCodeCircuit(distance=d, num_rounds=1)

    coords, stim_data, stim_ancilla, stim_support = _extract_stim_canonical_layout(d)
    rot = _rotate_45(coords)

    stim_to_qiskit_data = {s: i for i, s in enumerate(stim_data)}
    stim_anc_qsupport = {
        a: sorted(stim_to_qiskit_data[s] for s in stim_support[a])
        for a in stim_ancilla
    }

    qiskit_stabs = [sorted(s) for s in (sc.x_stabilizers + sc.z_stabilizers)]

    qiskit_anc_to_stim: Dict[int, int] = {}
    used: set = set()
    for anc_local, qstab in enumerate(qiskit_stabs):
        matched = None
        for sa in stim_ancilla:
            if sa in used:
                continue
            if stim_anc_qsupport[sa] == qstab:
                matched = sa
                break
        if matched is None:
            raise RuntimeError(
                f"[_build_logical_to_grid] No Stim ancilla matches Qiskit "
                f"stabilizer #{anc_local} (support {qstab}). "
                f"Stim supports (in Qiskit indices): {stim_anc_qsupport}"
            )
        qiskit_anc_to_stim[anc_local] = matched
        used.add(matched)

    logical_to_pos: Dict[int, Tuple[int, int]] = {}
    for qidx, sidx in enumerate(stim_data):
        logical_to_pos[qidx] = rot[sidx]
    for anc_local, sa in qiskit_anc_to_stim.items():
        logical_to_pos[d * d + anc_local] = rot[sa]

    cx_edges: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
    for anc_local, qstab in enumerate(qiskit_stabs):
        anc_logical = d * d + anc_local
        anc_pos = logical_to_pos[anc_logical]
        for dq in qstab:
            dq_pos = logical_to_pos[dq]
            dist = abs(anc_pos[0] - dq_pos[0]) + abs(anc_pos[1] - dq_pos[1])
            if dist != 1:
                raise AssertionError(
                    f"[_build_logical_to_grid] Non-4-neighbor CX edge: "
                    f"ancilla {anc_logical}@{anc_pos} <-> data {dq}@{dq_pos} "
                    f"(manhattan distance {dist}). "
                    f"45-degree rotation should yield all distance-1 edges."
                )
            cx_edges.append((anc_logical, anc_pos, dq_pos))

    rs = [p[0] for p in logical_to_pos.values()]
    cs = [p[1] for p in logical_to_pos.values()]
    patch_h = max(rs) - min(rs) + 1
    patch_w = max(cs) - min(cs) + 1
    expected = 2 * d - 1
    if patch_h != expected or patch_w != expected:
        raise AssertionError(
            f"[_build_logical_to_grid] Patch bounding box {patch_h}x{patch_w} "
            f"does not match expected {expected}x{expected} for distance={d}."
        )

    return logical_to_pos, cx_edges


def _patch_physical_map(logical_to_pos: Dict[int, Tuple[int, int]],
                        row_offset: int, col_offset: int,
                        ncols: int) -> Dict[int, int]:
    """Shift local grid positions by (row_offset, col_offset) and flatten."""
    return {
        log: (row_offset + lr) * ncols + (col_offset + lc)
        for log, (lr, lc) in logical_to_pos.items()
    }


def _get_edge_cx_error(properties, a: int, b: int) -> Optional[float]:
    """Read 2q gate error for edge (a,b) from BackendProperties. Returns None
    if absent."""
    try:
        # Qiskit BackendProperties exposes gate_error("cx"|"ecr"|"cz", qubits)
        for gate_name in ("ecr", "cz", "cx"):
            try:
                return properties.gate_error(gate_name, [a, b])
            except Exception:
                continue
        return None
    except Exception:
        return None


def _get_readout_error(properties, q: int) -> Optional[float]:
    try:
        return properties.readout_error(q)
    except Exception:
        return None


def select_best_patch(backend,
                      distance: int,
                      strategy: str = "min_cx_error",
                      grid_shape: Optional[Tuple[int, int]] = None,
                      verbose: bool = True) -> dict:
    """
    Pick a contiguous (2d-1) x (2d-1) patch on `backend` that can host the
    rotated surface code, minimizing average CX error across the patch's
    required stabilizer edges.

    Args:
        backend: a Qiskit BackendV2 with .coupling_map and .properties()
        distance: surface-code distance (3, 5, 7)
        strategy: "min_cx_error" (default) or "min_readout_error"
        grid_shape: (rows, cols) override; default inferred from num_qubits
        verbose: print diagnostics

    Returns:
        dict with keys:
            physical_qubits   -- {logical_idx: physical_idx}
            data_qubits       -- list of physical indices (length d^2)
            ancilla_qubits    -- list of physical indices
            initial_layout    -- list ordered by logical index
            diagnostics       -- {avg_cx_error, avg_readout_error, patch_origin,
                                  missing_edges, num_candidates}
    """
    d = distance
    try:
        num_qubits = backend.num_qubits
    except AttributeError:
        num_qubits = len(list(backend.properties().qubits))

    if grid_shape is None:
        grid_shape = _infer_grid_shape(num_qubits)
    nrows, ncols = grid_shape

    logical_to_pos, cx_grid_edges = _build_logical_to_grid(d)

    patch_h = max(p[0] for p in logical_to_pos.values()) + 1
    patch_w = max(p[1] for p in logical_to_pos.values()) + 1
    if patch_h > nrows or patch_w > ncols:
        raise ValueError(
            f"distance={d} patch ({patch_h}x{patch_w}) exceeds grid "
            f"({nrows}x{ncols})."
        )

    coupling = backend.coupling_map
    edge_list = list(coupling.get_edges()) if coupling is not None else []
    adj = _coupling_adjacency(edge_list)
    try:
        properties = backend.properties()
    except Exception:
        properties = None

    best = None
    num_candidates = 0
    for r0 in range(nrows - patch_h + 1):
        for c0 in range(ncols - patch_w + 1):
            phys_map = _patch_physical_map(logical_to_pos, r0, c0, ncols)

            missing = []
            edge_errors = []
            for anc_log, anc_pos, dq_pos in cx_grid_edges:
                anc_phys = (r0 + anc_pos[0]) * ncols + (c0 + anc_pos[1])
                dq_phys = (r0 + dq_pos[0]) * ncols + (c0 + dq_pos[1])
                if anc_phys == dq_phys:
                    missing.append((anc_phys, dq_phys))
                    continue
                if dq_phys not in adj.get(anc_phys, set()):
                    missing.append((anc_phys, dq_phys))
                    continue
                if properties is not None:
                    err = _get_edge_cx_error(properties, anc_phys, dq_phys)
                    if err is not None:
                        edge_errors.append(err)
            num_candidates += 1
            if missing:
                continue

            avg_cx = float(np.mean(edge_errors)) if edge_errors else float("nan")
            if properties is not None:
                ro_errors = [
                    r for q in phys_map.values()
                    for r in [_get_readout_error(properties, q)] if r is not None
                ]
                avg_ro = float(np.mean(ro_errors)) if ro_errors else float("nan")
            else:
                avg_ro = float("nan")

            score = avg_cx if strategy == "min_cx_error" else avg_ro
            if score != score:
                score = 0.0
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "origin": (r0, c0),
                    "phys_map": phys_map,
                    "avg_cx": avg_cx,
                    "avg_ro": avg_ro,
                }

    if best is None:
        raise RuntimeError(
            f"No valid {patch_h}x{patch_w} patch found on {grid_shape} grid. "
            f"Coupling map may not be row-major square lattice; consider "
            f"supplying grid_shape explicitly or implementing bipartite "
            f"fallback."
        )

    phys_map = best["phys_map"]
    data_qubits = [phys_map[i] for i in range(d * d)]
    ancilla_qubits = [phys_map[i] for i in range(d * d, len(phys_map))]
    initial_layout = [phys_map[i] for i in range(len(phys_map))]

    diagnostics = {
        "avg_cx_error": best["avg_cx"],
        "avg_readout_error": best["avg_ro"],
        "patch_origin": best["origin"],
        "grid_shape": grid_shape,
        "num_candidates": num_candidates,
        "strategy": strategy,
    }

    if verbose:
        print(f"[nighthawk_layout] distance={d}, grid={grid_shape}, "
              f"candidates={num_candidates}")
        print(f"    origin={best['origin']}, avg_cx_error={best['avg_cx']:.4e}, "
              f"avg_readout_error={best['avg_ro']:.4e}")
        print(f"    data_qubits={data_qubits}")
        print(f"    ancilla_qubits={ancilla_qubits}")

    return {
        "physical_qubits": phys_map,
        "data_qubits": data_qubits,
        "ancilla_qubits": ancilla_qubits,
        "initial_layout": initial_layout,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))

    class _CouplingMap:
        def __init__(self, edges):
            self._edges = edges

        def get_edges(self):
            return self._edges

    class _StubBackend:
        """4-neighbor row-major square lattice stub (matches Nighthawk topology)."""

        def __init__(self, rows, cols):
            self.num_qubits = rows * cols
            edges = []
            for r in range(rows):
                for c in range(cols):
                    q = r * cols + c
                    if c + 1 < cols:
                        edges.append((q, q + 1))
                    if r + 1 < rows:
                        edges.append((q, q + cols))
            self.coupling_map = _CouplingMap(edges)

        def properties(self):
            return None

    be = _StubBackend(*NIGHTHAWK_GRID_SHAPE)
    for d in (3, 5):
        print(f"\n=== distance={d} ===")
        res = select_best_patch(be, distance=d)
        print(f"    layout[0..4]={res['initial_layout'][:5]} ...")
