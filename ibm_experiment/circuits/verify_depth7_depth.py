#!/usr/bin/env python3
"""
Heavy-Hex Surface Code depth7 — Per-section depth verification.

heavyhex_surface_code_depth7.py 의 _depth7_round 가 정말 depth=7 인지
qc.barrier() 단위로 슬라이스해서 각 section 의 depth / CX-depth 를 측정.

기존 회로 코드는 수정하지 않으며, 이 파일은 단독 실행용.

사용:
    cd /home/stl/Seokhyeon_Projects/KCS
    python -m ibm_experiment.circuits.verify_depth7_depth          # 기본값 (no-DD, 1 cycle)
    python -m ibm_experiment.circuits.verify_depth7_depth --dd     # DD 켜기
    python -m ibm_experiment.circuits.verify_depth7_depth --cycles 3
    python -m ibm_experiment.circuits.verify_depth7_depth --transpile  # IBM backend 로 transpile 한 후 측정
"""

import argparse
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ibm_experiment.circuits.heavyhex_surface_code_depth7 import HeavyHexSurfaceCode


def _qubit_index(qc: QuantumCircuit, qubit) -> int:
    """qiskit 버전에 안전한 qubit -> int index 변환."""
    try:
        return qc.find_bit(qubit).index
    except AttributeError:
        return qc.qubits.index(qubit)


def _clbit_index(qc: QuantumCircuit, clbit) -> int:
    try:
        return qc.find_bit(clbit).index
    except AttributeError:
        return qc.clbits.index(clbit)


def slice_by_barrier(qc: QuantumCircuit):
    """
    qc 를 barrier 기준으로 잘라서 (header, sections...) 형태로 반환.

    반환: list[QuantumCircuit] — 각 element 가 한 section.
    각 section 은 같은 qubit/clbit 수를 갖는 새 QuantumCircuit.
    """
    dag = circuit_to_dag(qc)
    sections, current = [], []
    for node in dag.topological_op_nodes():
        if node.op.name == "barrier":
            if current:
                sections.append(current)
                current = []
        else:
            current.append(node)
    if current:
        sections.append(current)

    sub_circuits = []
    for sec in sections:
        sub = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        for n in sec:
            qargs = [_qubit_index(qc, q) for q in n.qargs]
            cargs = [_clbit_index(qc, c) for c in n.cargs]
            sub.append(n.op, qargs, cargs)
        sub_circuits.append(sub)
    return sub_circuits


def _two_qubit_filter(instr):
    """
    qc.depth(filter_function=...) 에 넘길 콜백.
    qiskit 버전에 따라 instr 가 CircuitInstruction 이거나 DAGOpNode 이므로 양쪽 케이스 처리.
    """
    op = getattr(instr, "operation", None) or getattr(instr, "op", None)
    if op is None:
        return False
    return op.num_qubits == 2


def report_sections(label: str, qc: QuantumCircuit):
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    total_ops = qc.count_ops()
    print(f"  whole circuit: depth={qc.depth()}, "
          f"CX-depth={qc.depth(_two_qubit_filter)}, "
          f"ops={dict(total_ops)}")
    print()

    sections = slice_by_barrier(qc)
    print(f"  --- {len(sections)} sections (split by qc.barrier()) ---")
    for i, sub in enumerate(sections):
        ops = dict(sub.count_ops())
        d = sub.depth()
        cx_d = sub.depth(_two_qubit_filter)
        n_cx = ops.get("cx", 0)
        n_meas = ops.get("measure", 0)
        n_h = ops.get("h", 0)
        n_reset = ops.get("reset", 0)
        flag = ""
        if d == 7:
            flag = "  <- depth-7 OK"
        elif d > 7:
            flag = f"  <- ⚠ depth={d} > 7"
        print(f"  Section {i:>2}: depth={d:>2}  CX-depth={cx_d:>2}  "
              f"cx={n_cx:>2}  h={n_h:>2}  meas={n_meas:>2}  reset={n_reset:>2}"
              f"{flag}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--dd", action="store_true",
                        help="manual X-Y DD 사용 (기본 off)")
    parser.add_argument("--init", type=int, default=0, choices=[0, 1])
    parser.add_argument("--transpile", action="store_true",
                        help="ibm fake backend 로 transpile 한 후 측정")
    parser.add_argument("--opt-level", type=int, default=1)
    args = parser.parse_args()

    sc = HeavyHexSurfaceCode(num_cycles=args.cycles, dd=args.dd)
    qc = sc.build_circuit(initial_state=args.init)

    label = (f"Heavy-Hex depth7 | cycles={args.cycles} dd={args.dd} "
             f"init={args.init} (logical circuit, pre-transpile)")
    report_sections(label, qc)

    if args.transpile:
        from qiskit import transpile
        try:
            from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
            backend = FakeSherbrooke()
            backend_name = "FakeSherbrooke"
        except Exception as e:
            print(f"[WARN] FakeSherbrooke 로드 실패 ({e}), AerSimulator 로 대체")
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            backend_name = "AerSimulator"

        qc_t = transpile(qc, backend=backend,
                         optimization_level=args.opt_level)
        label_t = (f"Heavy-Hex depth7 | cycles={args.cycles} dd={args.dd} "
                   f"init={args.init} | transpiled on {backend_name} "
                   f"(opt={args.opt_level})")
        report_sections(label_t, qc_t)


if __name__ == "__main__":
    main()
