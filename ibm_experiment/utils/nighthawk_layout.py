"""
Nighthawk (ibm_miami) qubit selection for rotated surface code.

ibm_miami is a ~120 qubit square lattice (best estimate: 10x12). The rotated
surface code embeds naturally: data qubits on even grid sites, ancilla on odd
sites. Each ancilla is adjacent (in the coupling graph) to the 2-4 data qubits
of its stabilizer.

This module enumerates (d+1)x(d+1) sliding-window candidate patches, validates
coupling-map coverage for all required CX edges of a given distance, then
scores each patch by average CX / readout error to pick the best.

Compatible logical indexing with `qiskit_surface_code_generator.SurfaceCodeCircuit`:
  - Data qubits: logical indices 0 .. d^2 - 1 (row-major in the dxd data grid)
  - Ancilla qubits: logical indices d^2 .. d^2 + num_stabilizers - 1
    (order matches SurfaceCodeCircuit.x_stabilizers + z_stabilizers)
"""

import os
import sys

import numpy as np
from typing import Dict, List, Tuple, Optional

_this_dir = os.path.dirname(os.path.abspath(__file__))
_ibm_dir = os.path.dirname(_this_dir)
if _ibm_dir not in sys.path:
    sys.path.insert(0, _ibm_dir)

from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit


def _infer_grid_shape(num_qubits: int) -> Tuple[int, int]:
    """Guess (rows, cols) for Nighthawk-style square lattice."""
    if num_qubits == 120:
        return (10, 12)
    # Try to find factor pair closest to square
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


def _build_logical_to_grid(distance: int) -> Tuple[Dict[int, Tuple[int, int]], List[Tuple[int, Tuple[int, int], Tuple[int, int]]]]:
    """
    Build the logical-qubit -> local grid-position mapping for the rotated
    surface code patch. Returns:
      logical_to_pos: dict mapping logical index -> (row, col) within a
        (2d-1) x (2d-1) local grid. Data sits at (2i, 2j); ancilla at the
        center of their stabilizer plaquette (or on the boundary edge).
      cx_edges: list of (ancilla_logical, (anc_pos), (data_pos)) tuples
        describing every CX adjacency the circuit requires, as grid positions.
    """
    d = distance
    sc = SurfaceCodeCircuit(distance=d, num_rounds=1)

    logical_to_pos: Dict[int, Tuple[int, int]] = {}
    # Data qubits on even sites: index r*d + c at (2r, 2c)
    for r in range(d):
        for c in range(d):
            logical_to_pos[r * d + c] = (2 * r, 2 * c)

    # Ancilla: stabilizer centers. Combined list: X stabs then Z stabs.
    all_stabs = sc.x_stabilizers + sc.z_stabilizers
    cx_edges: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
    for anc_local, stab in enumerate(all_stabs):
        anc_logical = d * d + anc_local
        positions = [logical_to_pos[q] for q in stab]
        ar = sum(p[0] for p in positions) / len(positions)
        ac = sum(p[1] for p in positions) / len(positions)
        # Round to nearest integer grid site; boundary-2 stabs have half-integer
        # coords which get rounded toward the interior.
        anc_pos = (int(round(ar)), int(round(ac)))
        # If collision with a data qubit (both coords even), shift along the
        # shorter extent toward the boundary.
        if anc_pos in logical_to_pos.values():
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]
            if min(rows) == max(rows):
                anc_pos = (rows[0] - 1 if rows[0] > 0 else rows[0] + 1, anc_pos[1])
            else:
                anc_pos = (anc_pos[0], cols[0] - 1 if cols[0] > 0 else cols[0] + 1)
        logical_to_pos[anc_logical] = anc_pos
        for dq in stab:
            cx_edges.append((anc_logical, anc_pos, logical_to_pos[dq]))

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

    class _StubBackend:
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
                    if r + 1 < rows and c + 1 < cols:
                        edges.append((q, q + cols + 1))
                    if r + 1 < rows and c > 0:
                        edges.append((q, q + cols - 1))

            class CM:
                def __init__(s, e): s._e = e
                def get_edges(s): return s._e
            self.coupling_map = CM(edges)

        def properties(self):
            return None

    be = _StubBackend(10, 12)
    for d in (3, 5):
        print(f"\n=== distance={d} ===")
        res = select_best_patch(be, distance=d)
        print(f"    layout[0..4]={res['initial_layout'][:5]} ...")
