#!/usr/bin/env python3
"""Reconstructed from project knowledge: ibm_experiment/circuits/heavyhex_surface_code_depth7.py"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List

DATA_QUBITS = [43, 45, 47, 63, 65, 67, 83, 85, 87]
LOGICAL_Z = [43, 63, 83]
LOGICAL_X = [43, 45, 47]

ROUND_DEFS = {
    1: {"A": 45, "br_AB": 44, "B": 43,
        "anc_Z": 56,
        "C": 63, "br_CD": 64, "D": 65,
        "anc_X": 77,
        "E": 85, "br_EF": 84, "F": 83,
        "bnd": [
            {"type": "Z", "qubits": (43, 45), "anc": 44, "name": "Zb_top"},
            {"type": "X", "qubits": (47, 67), "anc": 57, "name": "Xb_right"},
        ]},
    2: {"A": 45, "br_AB": 46, "B": 47,
        "anc_Z": 57,
        "C": 67, "br_CD": 66, "D": 65,
        "anc_X": 77,
        "E": 85, "br_EF": 86, "F": 87,
        "bnd": [
            {"type": "Z", "qubits": (85, 87), "anc": 86, "name": "Zb_bot"},
            {"type": "X", "qubits": (43, 63), "anc": 56, "name": "Xb_left"},
        ]},
}

ALL_PHYSICAL = sorted(set(
    DATA_QUBITS + [44, 46, 64, 66, 84, 86] + [56, 57, 77]
))


class HeavyHexSurfaceCode:
    def __init__(self, num_cycles: int = 1, dd: bool = True,
                 distance: int = 3, num_rounds: int = None,
                 backend_name: str = "ibm_boston"):
        if distance != 3:
            raise NotImplementedError("Only d=3 supported for heavy-hex embedding.")
        if num_rounds is not None:
            num_cycles = num_rounds

        self.num_cycles = num_cycles
        self.dd = dd
        self.distance = 3
        self.num_data = 9
        self._phys = ALL_PHYSICAL
        self._p2i = {p: i for i, p in enumerate(self._phys)}
        self._nq = len(self._phys)

        self.num_stabilizers = 8
        self.num_rounds = num_cycles
        self.backend_name = backend_name
        self._hw_syn_per_cycle = 8
        self._hw_syn_total = 8 * num_cycles + 2

    def q(self, phys: int) -> int:
        return self._p2i[phys]

    def _dd_idle(self, qc, active_set):
        if not self.dd:
            return
        for d in DATA_QUBITS:
            if d not in active_set:
                qc.x(self.q(d))
                qc.y(self.q(d))

    def build_circuit(self, initial_state: int = 0) -> QuantumCircuit:
        n_syn = 8 * self.num_cycles + 2
        qr = QuantumRegister(self._nq, "q")
        cr_syn = ClassicalRegister(n_syn, "syn")
        cr_dat = ClassicalRegister(self.num_data, "data")
        qc = QuantumCircuit(qr, cr_syn, cr_dat)

        if initial_state == 1:
            for d in LOGICAL_X:
                qc.x(self.q(d))

        syn_bit = 0
        prev_round = None

        for cycle in range(self.num_cycles):
            for rnd in [1, 2]:
                if prev_round is not None and prev_round != rnd:
                    syn_bit = self._transition_unfold(qc, prev_round, syn_bit, cr_syn)
                syn_bit = self._depth7_round(qc, rnd, syn_bit, cr_syn,
                                              first_ever=(prev_round is None))
                prev_round = rnd

        if prev_round is not None:
            syn_bit = self._final_unfold(qc, prev_round, syn_bit, cr_syn)

        for i, d in enumerate(DATA_QUBITS):
            qc.measure(self.q(d), cr_dat[i])

        return qc

    def _depth7_round(self, qc, rnd, syn_bit, cr_syn, first_ever=False):
        r = ROUND_DEFS[rnd]
        A, br_AB, B = r["A"], r["br_AB"], r["B"]
        anc_Z = r["anc_Z"]
        C, br_CD, D = r["C"], r["br_CD"], r["D"]
        anc_X = r["anc_X"]
        E, br_EF, F = r["E"], r["br_EF"], r["F"]

        # NO RESETS — key depth-7 feature

        # t1
        qc.cx(self.q(A), self.q(br_AB))
        qc.cx(self.q(D), self.q(br_CD))
        self._dd_idle(qc, {A, D, br_AB, br_CD})
        # t2
        qc.cx(self.q(br_AB), self.q(B))
        qc.cx(self.q(br_CD), self.q(C))
        self._dd_idle(qc, {B, C, br_AB, br_CD})
        # t3
        qc.cx(self.q(B), self.q(anc_Z))
        qc.cx(self.q(F), self.q(br_EF))
        self._dd_idle(qc, {B, F, anc_Z, br_EF})
        # t4
        qc.cx(self.q(C), self.q(anc_Z))
        qc.cx(self.q(br_EF), self.q(E))
        qc.h(self.q(anc_X))
        self._dd_idle(qc, {C, E, anc_Z, br_EF, anc_X})
        # t5
        qc.measure(self.q(anc_Z), cr_syn[syn_bit])
        syn_bit += 1
        qc.cx(self.q(anc_X), self.q(D))
        self._dd_idle(qc, {D, anc_Z, anc_X})
        # t6
        qc.cx(self.q(anc_X), self.q(E))
        self._dd_idle(qc, {E, anc_X})
        # t7
        qc.h(self.q(anc_X))
        qc.measure(self.q(anc_X), cr_syn[syn_bit])
        syn_bit += 1
        self._dd_idle(qc, {anc_X})

        for bnd in r["bnd"]:
            q0, q1 = bnd["qubits"]
            anc = bnd["anc"]
            if anc in (br_AB, br_CD, br_EF):
                continue
            if bnd["type"] == "Z":
                qc.cx(self.q(q0), self.q(anc))
                qc.cx(self.q(q1), self.q(anc))
            else:
                qc.h(self.q(anc))
                qc.cx(self.q(anc), self.q(q0))
                qc.cx(self.q(anc), self.q(q1))
                qc.h(self.q(anc))
            qc.measure(self.q(anc), cr_syn[syn_bit])
            syn_bit += 1

        qc.barrier()
        return syn_bit

    def _transition_unfold(self, qc, prev_rnd, syn_bit, cr_syn):
        r = ROUND_DEFS[prev_rnd]
        qc.cx(self.q(r["br_AB"]), self.q(r["B"]))
        qc.cx(self.q(r["A"]), self.q(r["br_AB"]))
        qc.cx(self.q(r["br_CD"]), self.q(r["C"]))
        qc.cx(self.q(r["D"]), self.q(r["br_CD"]))
        qc.cx(self.q(r["br_EF"]), self.q(r["E"]))
        qc.cx(self.q(r["F"]), self.q(r["br_EF"]))

        for bnd in r["bnd"]:
            anc = bnd["anc"]
            if anc not in (r["br_AB"], r["br_CD"], r["br_EF"]):
                continue
            q0, q1 = bnd["qubits"]
            qc.reset(self.q(anc))
            if bnd["type"] == "Z":
                qc.cx(self.q(q0), self.q(anc))
                qc.cx(self.q(q1), self.q(anc))
            else:
                qc.h(self.q(anc))
                qc.cx(self.q(anc), self.q(q0))
                qc.cx(self.q(anc), self.q(q1))
                qc.h(self.q(anc))
            qc.measure(self.q(anc), cr_syn[syn_bit])
            syn_bit += 1

        self._dd_idle(qc, {r["B"], r["C"], r["E"], r["A"], r["D"], r["F"],
                           r["br_AB"], r["br_CD"], r["br_EF"]})
        return syn_bit

    def _final_unfold(self, qc, last_rnd, syn_bit, cr_syn):
        r = ROUND_DEFS[last_rnd]
        qc.cx(self.q(r["br_AB"]), self.q(r["B"]))
        qc.cx(self.q(r["A"]), self.q(r["br_AB"]))
        qc.cx(self.q(r["br_CD"]), self.q(r["C"]))
        qc.cx(self.q(r["D"]), self.q(r["br_CD"]))
        qc.cx(self.q(r["br_EF"]), self.q(r["E"]))
        qc.cx(self.q(r["F"]), self.q(r["br_EF"]))

        for bnd in r["bnd"]:
            anc = bnd["anc"]
            if anc not in (r["br_AB"], r["br_CD"], r["br_EF"]):
                continue
            q0, q1 = bnd["qubits"]
            qc.reset(self.q(anc))
            if bnd["type"] == "Z":
                qc.cx(self.q(q0), self.q(anc))
                qc.cx(self.q(q1), self.q(anc))
            else:
                qc.h(self.q(anc))
                qc.cx(self.q(anc), self.q(q0))
                qc.cx(self.q(anc), self.q(q1))
                qc.h(self.q(anc))
            qc.measure(self.q(anc), cr_syn[syn_bit])
            syn_bit += 1
        return syn_bit
